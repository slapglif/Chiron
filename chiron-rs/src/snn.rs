//! Spiking neural network layers: PLIF neurons, SNN layer, temporal attention
//! and the MLP output head.
//!
//! CPU inference port of `chiron/layers/snn/model.py` (`PLIFNeuronLayer`,
//! `SNNLayer`, `TemporalAttention`, `MLPHead`). Training-only machinery
//! (surrogate gradients, gradient checkpointing, dropout) is omitted: spikes
//! are a hard Heaviside step and dropout is the identity.
//!
//! Optimizations relative to the Python reference:
//!
//! - **Precomputed PLIF decay constants.** The Python recomputes
//!   `beta_mem = exp(-dt / (1 + 49*sigmoid(p)))`, `beta_syn` and
//!   `V_th = softplus(p)` from their unconstrained parameters on *every*
//!   timestep. At inference the parameters are fixed, so `PlifNeuronLayer`
//!   caches `beta_mem`, `beta_syn` and `v_th` as per-neuron vectors at
//!   construction time.
//! - **Single-pass fused neuron state update.** The per-timestep elementwise
//!   updates (effective threshold, synaptic decay, membrane integration,
//!   spike, soft reset, adaptation) are fused into one rayon-parallel pass
//!   over rows, operating on contiguous slices. Only the recurrent drive
//!   (`mem @ W_rec^T`, a GEMM over the *previous* membrane state) runs
//!   separately, before the fused pass.
//! - **Constant input current hoisted out of the timestep loop**: `fc1(x)` is
//!   computed once and injected at every timestep, exactly as in the Python.
//!
//! Layer normalisation in `SnnLayer` is applied to the *spike* tensor
//! (post-spike normalisation), matching the Python: the normalised spikes —
//! not the raw binary spikes — feed `fc2` and the layer output.

use ndarray::{s, Array1, Array2, Array3, ArrayView2, ArrayView3, Axis, Zip};
use rand::rngs::StdRng;

use crate::nn::{gelu_inplace, orthogonal, sigmoid, softmax_slice, softplus, LayerNorm, Linear};

// ---------------------------------------------------------------------------
// PLIF defaults (mirroring PLIFNeuronLayer.__init__ in the Python).
// ---------------------------------------------------------------------------

const DT: f32 = 1.0;
const TAU_MEM_INIT: f32 = 10.0;
const TAU_SYN_INIT: f32 = 5.0;
const THRESHOLD_INIT: f32 = 1.0;
const ADAPT_BETA: f32 = 0.07;
const ADAPT_TAU: f32 = 100.0;
const TAU_MIN: f32 = 1.0;
const TAU_MAX: f32 = 50.0;

/// Inverse of the sigmoid parameterisation used to constrain tau:
/// `tau = tau_min + (tau_max - tau_min) * sigmoid(p)`.
#[inline]
fn inv_sigmoid_tau(tau: f32) -> f32 {
    let x = ((tau - TAU_MIN) / (TAU_MAX - TAU_MIN)).clamp(0.001, 0.999);
    (x / (1.0 - x)).ln()
}

#[inline]
fn tau_from_param(p: f32) -> f32 {
    TAU_MIN + (TAU_MAX - TAU_MIN) * sigmoid(p)
}

// ---------------------------------------------------------------------------
// PLIF neuron layer
// ---------------------------------------------------------------------------

/// Mutable neuron state carried across timesteps: membrane potential,
/// synaptic current and the adaptive-threshold variable, each `(n, size)`.
pub struct PlifState {
    pub mem: Array2<f32>,
    pub syn: Array2<f32>,
    pub adapt: Array2<f32>,
}

impl PlifState {
    pub fn zeros(n: usize, size: usize) -> Self {
        Self {
            mem: Array2::zeros((n, size)),
            syn: Array2::zeros((n, size)),
            adapt: Array2::zeros((n, size)),
        }
    }
}

/// Parametric leaky integrate-and-fire neuron population with adaptive
/// threshold (ALIF), soft reset, and optional recurrent connections.
///
/// All per-neuron constants (`beta_mem`, `beta_syn`, `v_th`) are precomputed
/// at construction — see the module docs.
pub struct PlifNeuronLayer {
    pub size: usize,
    recurrent: bool,
    adapt_beta: f32,
    /// `1 - dt / adapt_tau` (scalar).
    adapt_decay: f32,
    /// `exp(-dt / tau_mem)` per neuron.
    beta_mem: Array1<f32>,
    /// `exp(-dt / tau_syn)` per neuron.
    beta_syn: Array1<f32>,
    /// `softplus(v_th_param)` per neuron (base threshold, always positive).
    v_th: Array1<f32>,
    /// Recurrent weight, stored transposed (PyTorch `nn.Linear` computes
    /// `mem @ W^T`; we store `W^T` so the recurrent drive is `mem.dot(w_rec_t)`).
    w_rec_t: Option<Array2<f32>>,
}

impl PlifNeuronLayer {
    /// Build a PLIF population of `size` neurons with the Python defaults
    /// (dt=1, tau_mem=10, tau_syn=5, threshold=1, adapt_beta=0.07,
    /// adapt_tau=100). Recurrent weights are orthogonal with gain 0.5.
    pub fn new(size: usize, recurrent: bool, rng: &mut StdRng) -> Self {
        let beta_mem_v = (-DT / tau_from_param(inv_sigmoid_tau(TAU_MEM_INIT))).exp();
        let beta_syn_v = (-DT / tau_from_param(inv_sigmoid_tau(TAU_SYN_INIT))).exp();
        let v_th_v = softplus(THRESHOLD_INIT);
        let w_rec_t = if recurrent {
            // orthogonal() yields W; store W^T for `mem.dot(w_rec_t)`.
            Some(orthogonal(size, 0.5, rng).t().to_owned())
        } else {
            None
        };
        Self {
            size,
            recurrent,
            adapt_beta: ADAPT_BETA,
            adapt_decay: 1.0 - DT / ADAPT_TAU,
            beta_mem: Array1::from_elem(size, beta_mem_v),
            beta_syn: Array1::from_elem(size, beta_syn_v),
            v_th: Array1::from_elem(size, v_th_v),
            w_rec_t,
        }
    }

    /// Simulate one timestep. Returns binary spikes (0.0 / 1.0) of shape
    /// `(n, size)` and updates `state` in place.
    ///
    /// Update order (matches the Python exactly):
    /// ```text
    /// v_th_eff = v_th + adapt_beta * adapt          (uses OLD adapt)
    /// syn      = beta_syn * syn + input
    /// rec      = mem_old @ W_rec^T                  (uses OLD mem)
    /// mem      = beta_mem * mem + syn + rec
    /// spike    = (mem > v_th_eff)
    /// mem     -= v_th_eff * spike                   (soft reset)
    /// adapt    = adapt_decay * adapt + spike
    /// ```
    pub fn step(&self, input_current: &ArrayView2<f32>, state: &mut PlifState) -> Array2<f32> {
        let (n, size) = input_current.dim();
        debug_assert_eq!(size, self.size);
        debug_assert_eq!(state.mem.dim(), (n, size));

        // Recurrent drive uses the PREVIOUS membrane potential, so the GEMM
        // must run before the fused state update below.
        let rec = self.w_rec_t.as_ref().map(|w| state.mem.dot(w));

        let mut spikes = Array2::<f32>::zeros((n, size));
        let beta_mem = self.beta_mem.as_slice().unwrap();
        let beta_syn = self.beta_syn.as_slice().unwrap();
        let v_th = self.v_th.as_slice().unwrap();
        let adapt_beta = self.adapt_beta;
        let adapt_decay = self.adapt_decay;

        // Fused elementwise update over one contiguous row.
        #[inline]
        #[allow(clippy::too_many_arguments)] // flat slice args keep the hot loop allocation-free
        fn update_row(
            sp: &mut [f32],
            mem: &mut [f32],
            syn: &mut [f32],
            adapt: &mut [f32],
            inp: &[f32],
            rec: Option<&[f32]>,
            beta_mem: &[f32],
            beta_syn: &[f32],
            v_th: &[f32],
            adapt_beta: f32,
            adapt_decay: f32,
        ) {
            for j in 0..sp.len() {
                let v_th_eff = v_th[j] + adapt_beta * adapt[j];
                syn[j] = beta_syn[j] * syn[j] + inp[j];
                let r = match rec {
                    Some(r) => r[j],
                    None => 0.0,
                };
                mem[j] = beta_mem[j] * mem[j] + syn[j] + r;
                let spike = if mem[j] > v_th_eff { 1.0 } else { 0.0 };
                mem[j] -= v_th_eff * spike; // soft reset
                adapt[j] = adapt_decay * adapt[j] + spike;
                sp[j] = spike;
            }
        }

        match &rec {
            Some(rec) => {
                Zip::from(spikes.rows_mut())
                    .and(state.mem.rows_mut())
                    .and(state.syn.rows_mut())
                    .and(state.adapt.rows_mut())
                    .and(input_current.rows())
                    .and(rec.rows())
                    .par_for_each(|mut sp, mut mem, mut syn, mut adapt, inp, rc| {
                        update_row(
                            sp.as_slice_mut().unwrap(),
                            mem.as_slice_mut().unwrap(),
                            syn.as_slice_mut().unwrap(),
                            adapt.as_slice_mut().unwrap(),
                            inp.as_slice().unwrap(),
                            Some(rc.as_slice().unwrap()),
                            beta_mem,
                            beta_syn,
                            v_th,
                            adapt_beta,
                            adapt_decay,
                        );
                    });
            }
            None => {
                Zip::from(spikes.rows_mut())
                    .and(state.mem.rows_mut())
                    .and(state.syn.rows_mut())
                    .and(state.adapt.rows_mut())
                    .and(input_current.rows())
                    .par_for_each(|mut sp, mut mem, mut syn, mut adapt, inp| {
                        update_row(
                            sp.as_slice_mut().unwrap(),
                            mem.as_slice_mut().unwrap(),
                            syn.as_slice_mut().unwrap(),
                            adapt.as_slice_mut().unwrap(),
                            inp.as_slice().unwrap(),
                            None,
                            beta_mem,
                            beta_syn,
                            v_th,
                            adapt_beta,
                            adapt_decay,
                        );
                    });
            }
        }
        spikes
    }

    /// Whether this population has recurrent connections.
    #[inline]
    pub fn is_recurrent(&self) -> bool {
        self.recurrent
    }
}

// ---------------------------------------------------------------------------
// SNN layer: fc1 -> PLIF -> LN -> fc2 -> PLIF -> LN over T timesteps
// ---------------------------------------------------------------------------

/// Two-population spiking layer driven by a constant input current for
/// `timesteps` steps. Dropout is the identity (inference).
pub struct SnnLayer {
    pub fc1: Linear,
    pub fc2: Linear,
    pub neurons1: PlifNeuronLayer,
    pub neurons2: PlifNeuronLayer,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub timesteps: usize,
}

impl SnnLayer {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        timesteps: usize,
        rng: &mut StdRng,
    ) -> Self {
        Self {
            fc1: Linear::new(input_size, hidden_size, true, rng),
            fc2: Linear::new(hidden_size, output_size, true, rng),
            neurons1: PlifNeuronLayer::new(hidden_size, true, rng),
            neurons2: PlifNeuronLayer::new(output_size, true, rng),
            ln1: LayerNorm::new(hidden_size),
            ln2: LayerNorm::new(output_size),
            input_size,
            hidden_size,
            output_size,
            timesteps,
        }
    }

    /// `x: (N = batch*seq, input_size) -> (N, timesteps, output_size)`.
    ///
    /// `fc1(x)` is computed once and injected as a constant current at every
    /// timestep. LayerNorm is applied to the spike tensors (the normalised —
    /// no longer binary — spikes feed `fc2` and the output), exactly as in
    /// the Python.
    pub fn forward(&self, x: &ArrayView2<f32>) -> Array3<f32> {
        let n = x.nrows();
        debug_assert_eq!(x.ncols(), self.input_size);

        let current1 = self.fc1.forward(x); // constant current, computed once
        let mut state1 = PlifState::zeros(n, self.hidden_size);
        let mut state2 = PlifState::zeros(n, self.output_size);
        let mut out = Array3::<f32>::zeros((n, self.timesteps, self.output_size));

        for t in 0..self.timesteps {
            let mut spike1 = self.neurons1.step(&current1.view(), &mut state1);
            self.ln1.forward_inplace(spike1.view_mut());
            let current2 = self.fc2.forward(&spike1.view());
            let mut spike2 = self.neurons2.step(&current2.view(), &mut state2);
            self.ln2.forward_inplace(spike2.view_mut());
            out.slice_mut(s![.., t, ..]).assign(&spike2);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Temporal attention over timesteps
// ---------------------------------------------------------------------------

/// Single-head attention that collapses the timestep axis: the query is the
/// timestep mean, the keys are the per-timestep features.
///
/// `x: (N, T, F) -> (N, F)`.
pub struct TemporalAttention {
    pub query: Linear,
    pub key: Linear,
    pub scale: f32,
    pub ln: LayerNorm,
}

impl TemporalAttention {
    pub fn new(feat_dim: usize, rng: &mut StdRng) -> Self {
        Self {
            query: Linear::new(feat_dim, feat_dim, true, rng),
            key: Linear::new(feat_dim, feat_dim, true, rng),
            scale: (feat_dim as f32).sqrt(),
            ln: LayerNorm::new(feat_dim),
        }
    }

    /// `x: (N, T, F) -> (N, F)`. Scores are `dot(q[n], k[n,t]) / sqrt(F)`,
    /// softmaxed over T; the output is the weighted timestep sum, layer-normed.
    pub fn forward(&self, x: &ArrayView3<f32>) -> Array2<f32> {
        let (n, t, f) = x.dim();

        // q = query(mean over T): (N, F). Two GEMMs total.
        let mean = x.mean_axis(Axis(1)).expect("T must be >= 1");
        let q = self.query.forward(&mean.view());
        let x_flat = x.to_shape((n * t, f)).expect("reshape (N*T, F)");
        let k = self.key.forward(&x_flat.view());
        let k3 = k
            .into_shape_with_order((n, t, f))
            .expect("reshape back (N, T, F)");

        let scale = self.scale;
        let mut out = Array2::<f32>::zeros((n, f));
        Zip::from(out.rows_mut())
            .and(q.rows())
            .and(k3.axis_iter(Axis(0)))
            .and(x.axis_iter(Axis(0)))
            .par_for_each(|mut o, qr, kn, xn| {
                // scores over T, softmax, then weighted sum of timesteps.
                let mut scores: Vec<f32> =
                    (0..t).map(|ti| qr.dot(&kn.row(ti)) / scale).collect();
                softmax_slice(&mut scores);
                for (ti, &w) in scores.iter().enumerate() {
                    let xr = xn.row(ti);
                    for (ov, &xv) in o.iter_mut().zip(xr.iter()) {
                        *ov += w * xv;
                    }
                }
            });
        self.ln.forward_inplace(out.view_mut());
        out
    }
}

// ---------------------------------------------------------------------------
// MLP output head
// ---------------------------------------------------------------------------

/// Two-layer MLP: `fc1 -> LayerNorm -> GELU -> fc2` (dropout is identity).
pub struct MlpHead {
    pub fc1: Linear,
    pub ln: LayerNorm,
    pub fc2: Linear,
}

impl MlpHead {
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        rng: &mut StdRng,
    ) -> Self {
        Self {
            fc1: Linear::new(in_features, hidden_features, true, rng),
            ln: LayerNorm::new(hidden_features),
            fc2: Linear::new(hidden_features, out_features, true, rng),
        }
    }

    /// `x: (N, in) -> (N, out)`.
    pub fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let mut h = self.fc1.forward(x);
        self.ln.forward_inplace(h.view_mut());
        gelu_inplace(&mut h);
        self.fc2.forward(&h.view())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::seeded_rng;

    fn det_input(rows: usize, cols: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |(i, j)| {
            ((i * 7 + j * 3) as f32 * 0.37).sin()
        })
    }

    #[test]
    fn plif_spikes_are_binary() {
        let mut rng = seeded_rng(1);
        let layer = PlifNeuronLayer::new(16, true, &mut rng);
        let mut state = PlifState::zeros(5, 16);
        // Mixed positive / negative drive.
        let input = det_input(5, 16) * 3.0;
        for _ in 0..8 {
            let spikes = layer.step(&input.view(), &mut state);
            assert_eq!(spikes.dim(), (5, 16));
            for &v in spikes.iter() {
                assert!(v == 0.0 || v == 1.0, "non-binary spike value {v}");
            }
        }
    }

    #[test]
    fn plif_soft_reset_keeps_membrane_bounded() {
        let mut rng = seeded_rng(2);
        let layer = PlifNeuronLayer::new(4, false, &mut rng);
        let mut state = PlifState::zeros(1, 4);
        let input = Array2::from_elem((1, 4), 5.0_f32);
        let mut total_spikes = 0.0_f32;
        for _ in 0..200 {
            total_spikes += layer.step(&input.view(), &mut state).sum();
        }
        assert!(total_spikes > 0.0, "large constant drive must elicit spikes");
        for &m in state.mem.iter() {
            assert!(m.is_finite());
            // Without the soft reset, mem would converge near syn/(1-beta_mem)
            // plus unbounded threshold-free accumulation; reset keeps it small.
            assert!(m.abs() < 1000.0, "membrane potential unbounded: {m}");
        }
    }

    #[test]
    fn plif_adaptation_slows_firing() {
        let mut rng = seeded_rng(3);
        let layer = PlifNeuronLayer::new(1, false, &mut rng);
        let mut state = PlifState::zeros(1, 1);
        let input = Array2::from_elem((1, 1), 0.5_f32);
        let mut first_half = 0.0_f32;
        let mut second_half = 0.0_f32;
        for step in 0..200 {
            let s = layer.step(&input.view(), &mut state).sum();
            if step < 100 {
                first_half += s;
            } else {
                second_half += s;
            }
        }
        assert!(first_half >= 1.0, "supra-threshold drive must elicit spikes");
        assert!(
            second_half < first_half,
            "adaptive threshold should suppress tonic firing \
             (first={first_half}, second={second_half})"
        );
    }

    #[test]
    fn plif_no_input_no_spikes() {
        let mut rng = seeded_rng(4);
        let layer = PlifNeuronLayer::new(8, true, &mut rng);
        let mut state = PlifState::zeros(3, 8);
        let input = Array2::<f32>::zeros((3, 8));
        for _ in 0..20 {
            let spikes = layer.step(&input.view(), &mut state);
            assert_eq!(spikes.sum(), 0.0);
        }
    }

    #[test]
    fn snn_layer_output_shape() {
        let mut rng = seeded_rng(42);
        let layer = SnnLayer::new(12, 16, 8, 5, &mut rng);
        let x = det_input(6, 12);
        let y = layer.forward(&x.view());
        assert_eq!(y.dim(), (6, 5, 8));
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn snn_layer_deterministic_given_seed() {
        let x = det_input(4, 12);
        let mut rng_a = seeded_rng(42);
        let mut rng_b = seeded_rng(42);
        let a = SnnLayer::new(12, 16, 8, 4, &mut rng_a);
        let b = SnnLayer::new(12, 16, 8, 4, &mut rng_b);
        let ya = a.forward(&x.view());
        let yb = b.forward(&x.view());
        assert_eq!(ya, yb);
        // Re-running the same layer must also be reproducible.
        assert_eq!(ya, a.forward(&x.view()));
    }

    #[test]
    fn temporal_attention_shape_and_t1() {
        let mut rng = seeded_rng(5);
        let attn = TemporalAttention::new(8, &mut rng);

        let x = Array3::from_shape_fn((4, 3, 8), |(i, t, j)| {
            ((i * 11 + t * 5 + j) as f32 * 0.21).cos()
        });
        let y = attn.forward(&x.view());
        assert_eq!(y.dim(), (4, 8));
        assert!(y.iter().all(|v| v.is_finite()));

        // T = 1: softmax over a single timestep must not blow up
        // (weight = 1 exactly).
        let x1 = Array3::from_shape_fn((4, 1, 8), |(i, _, j)| ((i + j) as f32 * 0.3).sin());
        let y1 = attn.forward(&x1.view());
        assert_eq!(y1.dim(), (4, 8));
        assert!(y1.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mlp_head_shape() {
        let mut rng = seeded_rng(6);
        let head = MlpHead::new(8, 6, 10, &mut rng);
        let x = det_input(5, 8);
        let y = head.forward(&x.view());
        assert_eq!(y.dim(), (5, 10));
        assert!(y.iter().all(|v| v.is_finite()));
    }
}
