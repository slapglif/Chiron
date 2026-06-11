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
//! - **Event-driven sparse path** ([`SnnLayer::forward_events`] /
//!   [`TemporalAttention::forward_events`] / [`SpikeTrain`]): spikes are
//!   binary, so LayerNorm of a spike row takes only *two* distinct values,
//!   fully determined by the spike count `k`. Every downstream
//!   `Linear(LayerNorm(spikes))` therefore collapses to a sparse accumulation
//!   of pre-scaled weight rows plus a rank-1 correction from two precomputable
//!   vectors — exact (not an approximation), see [`SpikeTrain::ln_scalars`].
//! - **Optional recurrent-weight pruning**
//!   ([`PlifNeuronLayer::prune_recurrent`]): a controllable approximation
//!   that replaces the dense recurrent GEMM with a CSR sparse matvec.
//!
//! Layer normalisation in `SnnLayer` is applied to the *spike* tensor
//! (post-spike normalisation), matching the Python: the normalised spikes —
//! not the raw binary spikes — feed `fc2` and the layer output.

use ndarray::{s, Array1, Array2, Array3, ArrayView2, ArrayView3, Axis, Zip};
use rand::rngs::StdRng;
use rayon::prelude::*;

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

/// CSR-like pruned recurrent weights, rows indexed by **output** neuron.
///
/// Row `j` holds the `(input_index, weight)` pairs that survived magnitude
/// pruning, so the recurrent drive is the sparse matvec
/// `out[n][j] = sum over stored pairs of mem[n][i] * w` — the sparse
/// counterpart of the dense `mem @ W_rec^T` GEMM.
struct RecurrentCsr {
    /// `len == size + 1`; row `j` spans `row_ptr[j]..row_ptr[j + 1]`.
    row_ptr: Vec<usize>,
    /// Input (presynaptic) neuron index per kept entry.
    col_idx: Vec<u32>,
    /// Kept weight `W_rec[j][i]` per entry.
    vals: Vec<f32>,
}

impl RecurrentCsr {
    /// Sparse recurrent drive `mem @ W_rec^T` using only the kept entries.
    /// Rayon-parallel over batch rows; the inner loops walk contiguous CSR
    /// slices with no allocation.
    fn matvec(&self, mem: &Array2<f32>) -> Array2<f32> {
        let n = mem.nrows();
        let size = self.row_ptr.len() - 1;
        let mut out = Array2::<f32>::zeros((n, size));
        Zip::from(out.rows_mut())
            .and(mem.rows())
            .par_for_each(|mut o, m| {
                let o = o.as_slice_mut().unwrap();
                let m = m.as_slice().unwrap();
                for (j, oj) in o.iter_mut().enumerate() {
                    let (lo, hi) = (self.row_ptr[j], self.row_ptr[j + 1]);
                    let mut acc = 0.0_f32;
                    for (&i, &w) in self.col_idx[lo..hi].iter().zip(&self.vals[lo..hi]) {
                        acc += m[i as usize] * w;
                    }
                    *oj = acc;
                }
            });
        out
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
    /// Magnitude-pruned recurrent weights; when present, `step` uses the
    /// sparse matvec instead of the dense GEMM. See [`Self::prune_recurrent`].
    w_rec_csr: Option<RecurrentCsr>,
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
            w_rec_csr: None,
        }
    }

    /// Prune the recurrent weight matrix to its top `keep` fraction of
    /// entries by absolute magnitude (`0 < keep <= 1.0`), globally across the
    /// whole matrix. This is a **controllable approximation**: the recurrent
    /// membrane GEMM is the dominant remaining per-timestep cost, and the
    /// kept entries are stored as a CSR structure over output neurons (see
    /// [`RecurrentCsr`]) so `step` runs a sparse matvec instead.
    ///
    /// - `keep == 1.0` clears the pruned structure and restores the exact
    ///   dense GEMM path.
    /// - Deterministic: magnitude ties are broken by linear entry index
    ///   `(output * size + input)`, ascending.
    /// - No-op on non-recurrent populations.
    pub fn prune_recurrent(&mut self, keep: f32) {
        assert!(
            keep > 0.0 && keep <= 1.0,
            "keep fraction must be in (0, 1], got {keep}"
        );
        if keep >= 1.0 {
            self.w_rec_csr = None;
            return;
        }
        let Some(wt) = &self.w_rec_t else {
            return; // non-recurrent population: nothing to prune
        };
        let size = self.size;
        let total = size * size;
        let kept = (((total as f64) * f64::from(keep)).round() as usize).clamp(1, total);

        // Linear index idx = j * size + i over (output j, input i); the
        // weight of entry idx is W_rec[j][i] = w_rec_t[(i, j)].
        let weight_at = |idx: usize| wt[(idx % size, idx / size)];
        let mut order: Vec<usize> = (0..total).collect();
        order.sort_unstable_by(|&a, &b| {
            weight_at(b)
                .abs()
                .total_cmp(&weight_at(a).abs())
                .then_with(|| a.cmp(&b))
        });
        order.truncate(kept);
        order.sort_unstable(); // back to (j, i) order so rows come out grouped

        let mut row_ptr = vec![0_usize; size + 1];
        let mut col_idx = Vec::with_capacity(kept);
        let mut vals = Vec::with_capacity(kept);
        for &idx in &order {
            let (j, i) = (idx / size, idx % size);
            row_ptr[j + 1] += 1;
            col_idx.push(i as u32);
            vals.push(wt[(i, j)]);
        }
        for j in 0..size {
            row_ptr[j + 1] += row_ptr[j];
        }
        self.w_rec_csr = Some(RecurrentCsr {
            row_ptr,
            col_idx,
            vals,
        });
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
        // (or pruned sparse matvec) must run before the fused state update.
        let rec = match (&self.w_rec_csr, &self.w_rec_t) {
            (Some(csr), _) => Some(csr.matvec(&state.mem)),
            (None, Some(w)) => Some(state.mem.dot(w)),
            (None, None) => None,
        };

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
// Bit-packed spike train
// ---------------------------------------------------------------------------

/// Call `f(bit_index)` for every set bit in `mask` (a row of `u64` words),
/// using the classic clear-lowest-set-bit trick: `w &= w - 1`.
#[inline]
fn for_each_set_bit(mask: &[u64], mut f: impl FnMut(usize)) {
    for (wi, &word) in mask.iter().enumerate() {
        let mut w = word;
        while w != 0 {
            let b = w.trailing_zeros() as usize;
            f(wi * 64 + b);
            w &= w - 1;
        }
    }
}

/// Bit-packed binary spike output of an [`SnnLayer`] across timesteps.
///
/// Layout of `masks` is `[n][t][word]`: row `n`, timestep `t` occupies
/// `words_per_row` consecutive `u64` words; bit `i` of the row is
/// `masks[(n*timesteps + t)*words_per_row + i/64] >> (i%64) & 1`.
///
/// **Why no floats are stored.** Spikes are binary `{0, 1}`, so LayerNorm of
/// a spike row takes only *two* distinct values, fully determined by the
/// popcount `k` of the row: with `m = k/dim` the row mean, the variance is
/// exactly `m * (1 - m)` (because `s_i^2 = s_i` for binary `s_i`), hence
/// `ln(s)_i = (s_i - m) * inv_std * gamma_i + beta_i` with
/// `inv_std = 1/sqrt(m(1-m) + eps)`. Everything a consumer needs derives
/// from the popcount via [`SpikeTrain::ln_scalars`]; the train itself stores
/// one bit per neuron per timestep ([`SpikeTrain::memory_bytes`]).
pub struct SpikeTrain {
    /// Number of rows (`batch * seq`).
    pub n: usize,
    /// Number of simulated timesteps `T`.
    pub timesteps: usize,
    /// Neurons per row (e.g. 128 at the SNN output layer).
    pub dim: usize,
    /// `ceil(dim / 64)` — `u64` words per `(n, t)` row.
    pub words_per_row: usize,
    /// Bit-packed spikes, layout `[n][t][word]`.
    pub masks: Vec<u64>,
}

impl SpikeTrain {
    /// The `words_per_row` packed words of row `n` at timestep `t`.
    #[inline]
    pub fn mask_row(&self, n: usize, t: usize) -> &[u64] {
        let start = (n * self.timesteps + t) * self.words_per_row;
        &self.masks[start..start + self.words_per_row]
    }

    /// Spike count `k` of row `n` at timestep `t`.
    #[inline]
    pub fn popcount_row(&self, n: usize, t: usize) -> u32 {
        self.mask_row(n, t).iter().map(|w| w.count_ones()).sum()
    }

    /// LayerNorm scalars `(m, inv_std)` of a binary row with `k` set bits out
    /// of `dim`, for a LayerNorm with the given `eps`.
    ///
    /// For binary `s` the mean is `m = k/dim` and, since `s_i^2 = s_i`, the
    /// (biased, as used by LayerNorm) variance is *exactly*
    /// `mean(s^2) - m^2 = m - m^2 = m(1 - m)`; thus
    /// `inv_std = 1/sqrt(m(1-m) + eps)` and the normalised row is the
    /// two-valued `ln(s)_i = (s_i - m) * inv_std * gamma_i + beta_i`.
    /// This identity is exact — only float summation order differs from the
    /// dense LayerNorm.
    #[inline]
    pub fn ln_scalars(k: u32, dim: usize, eps: f32) -> (f32, f32) {
        let m = k as f32 / dim as f32;
        let var = m * (1.0 - m);
        (m, 1.0 / (var + eps).sqrt())
    }

    /// Heap memory of the packed masks: `n * timesteps * words_per_row * 8`
    /// bytes, i.e. ~`dim/8` bytes per `(n, t)` row versus `4 * dim` for the
    /// dense `f32` cube.
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.masks.len() * std::mem::size_of::<u64>()
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

    /// Event-driven forward: produces the same spikes as [`Self::forward`]
    /// without ever materialising the post-`ln1` float matrix or the
    /// post-`ln2` float cube.
    ///
    /// `fc2(ln1(spike1))` is computed via the exact binary-LayerNorm identity
    /// (see [`SpikeTrain::ln_scalars`]). For a spike row `s` with popcount
    /// `k`, mean `m = k/hidden` and `inv_std = 1/sqrt(m(1-m) + eps)`:
    ///
    /// ```text
    /// W^T ln1(s) = inv_std * sum_{i in spikes} (gamma1_i * W[i,:])   sparse rows
    ///            + (-m * inv_std) * (gamma1^T W)                     rank-1
    ///            + (beta1^T W) + bias                                constant
    /// ```
    ///
    /// The gamma-scaled rows of `fc2.weight` and the two rank-1 vectors
    /// `gamma1^T W` / `beta1^T W` are precomputed once per call (one small
    /// GEMV pair plus a scaled copy). Per row only the spiking weight rows
    /// are accumulated — at low spike density this replaces the dense
    /// `hidden x out` GEMM with `~k` row adds. The result is *exact* up to
    /// float summation order. Output spikes (post-`neurons2`, **pre-`ln2`**,
    /// binary) are bit-packed into a [`SpikeTrain`]; consumers reconstruct
    /// the `ln2` values from popcounts plus [`Self::ln2`]'s `gamma`/`beta`.
    pub fn forward_events(&self, x: &ArrayView2<f32>) -> SpikeTrain {
        let n = x.nrows();
        debug_assert_eq!(x.ncols(), self.input_size);
        let hidden = self.hidden_size;
        let out_dim = self.output_size;

        // Fused ln1 -> fc2 operands (precomputed once per call).
        let w2 = &self.fc2.weight; // (hidden, out)
        let gamma1 = &self.ln1.gamma;
        // Row i = gamma1[i] * fc2.weight.row(i).
        let wg = Array2::from_shape_fn((hidden, out_dim), |(i, o)| gamma1[i] * w2[(i, o)]);
        let gw = gamma1.dot(w2); // gamma1^T W   (len out)
        let mut base = self.ln1.beta.dot(w2); // beta1^T W (len out)
        if let Some(b) = &self.fc2.bias {
            base += b;
        }
        let wg_s = wg.as_slice().unwrap();
        let gw_s = gw.as_slice().unwrap();
        let base_s = base.as_slice().unwrap();
        let eps1 = self.ln1.eps;

        let current1 = self.fc1.forward(x); // constant current, computed once
        let mut state1 = PlifState::zeros(n, hidden);
        let mut state2 = PlifState::zeros(n, out_dim);
        let mut current2 = Array2::<f32>::zeros((n, out_dim)); // reused buffer

        let wpr = out_dim.div_ceil(64);
        let mut masks = vec![0_u64; n * self.timesteps * wpr];
        let row_words = self.timesteps * wpr;

        for t in 0..self.timesteps {
            let spike1 = self.neurons1.step(&current1.view(), &mut state1);

            // current2 = fc2(ln1(spike1)) via the sparse identity above.
            Zip::from(current2.rows_mut())
                .and(spike1.rows())
                .par_for_each(|mut c2, s1| {
                    let c2 = c2.as_slice_mut().unwrap();
                    let s1 = s1.as_slice().unwrap();
                    c2.fill(0.0);
                    // Accumulate gamma-scaled rows of spiking neurons,
                    // counting spikes in the same pass.
                    let mut k = 0_u32;
                    for (i, &v) in s1.iter().enumerate() {
                        if v != 0.0 {
                            k += 1;
                            let row = &wg_s[i * out_dim..(i + 1) * out_dim];
                            for (acc, &w) in c2.iter_mut().zip(row) {
                                *acc += w;
                            }
                        }
                    }
                    // Scale by inv_std and add the rank-1 + constant terms.
                    let (m, inv_std) = SpikeTrain::ln_scalars(k, hidden, eps1);
                    let c = -m * inv_std;
                    for ((acc, &g), &b) in c2.iter_mut().zip(gw_s).zip(base_s) {
                        *acc = *acc * inv_std + c * g + b;
                    }
                });

            let spike2 = self.neurons2.step(&current2.view(), &mut state2);

            // Bit-pack the binary spike2 rows at timestep t (no ln2 here —
            // the consumer reconstructs ln2 values from the popcount).
            masks
                .par_chunks_mut(row_words)
                .zip(spike2.axis_iter(Axis(0)).into_par_iter())
                .for_each(|(mrow, srow)| {
                    let dst = &mut mrow[t * wpr..(t + 1) * wpr];
                    for (j, &v) in srow.as_slice().unwrap().iter().enumerate() {
                        if v != 0.0 {
                            dst[j / 64] |= 1_u64 << (j % 64);
                        }
                    }
                });
        }

        SpikeTrain {
            n,
            timesteps: self.timesteps,
            dim: out_dim,
            words_per_row: wpr,
            masks,
        }
    }

    /// Prune the recurrent weights of **both** neuron populations to their
    /// top `keep` fraction by magnitude. `keep == 1.0` restores the exact
    /// dense path. See [`PlifNeuronLayer::prune_recurrent`].
    pub fn prune_recurrent(&mut self, keep: f32) {
        self.neurons1.prune_recurrent(keep);
        self.neurons2.prune_recurrent(keep);
    }

    /// The output-side LayerNorm, whose `gamma`/`beta`/`eps` consumers of a
    /// [`SpikeTrain`] need to reconstruct the post-`ln2` values from binary
    /// spikes (e.g. [`TemporalAttention::forward_events`]).
    #[inline]
    pub fn ln2(&self) -> &LayerNorm {
        &self.ln2
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

    /// Event-driven attention over a bit-packed [`SpikeTrain`], numerically
    /// matching [`Self::forward`] on the materialised `ln2` cube (identical
    /// up to float summation order).
    ///
    /// The dense input the old path used is `x[n,t]_i = (s_i - m_t) *
    /// inv_std_t * gamma_i + beta_i` with `(m_t, inv_std_t)` derived from the
    /// row popcount ([`SpikeTrain::ln_scalars`]) — the binary-LayerNorm
    /// two-value identity. Every dense op then decomposes into a *sparse
    /// accumulation over set bits plus rank-1 corrections*:
    ///
    /// - **timestep mean** `(1/T) sum_t x[n,t]`: sparse adds of
    ///   `inv_std_t * gamma_i` over set bits, plus
    ///   `(sum_t -m_t inv_std_t) * gamma` and `T * beta`, all divided by `T`;
    /// - **key scores** `dot(q_n, key(x[n,t]))`: with gamma-scaled key rows
    ///   `kg`, `gw_k = gamma^T W_k` and `bw_k = beta^T W_k + bias_k`
    ///   precomputed once per call, `qg = q kg^T`, `c1 = q gw_k`,
    ///   `c2 = q bw_k` reduce each score to a popcount-driven sum of
    ///   `qg[n,i]` over set bits — no per-timestep key vectors are ever
    ///   materialised;
    /// - **weighted timestep sum** `sum_t w_t x[n,t]`: sparse adds of
    ///   `w_t inv_std_t gamma_i`, plus `(sum_t -w_t m_t inv_std_t) * gamma`
    ///   and (softmax weights summing to 1) a single `+ beta`.
    ///
    /// Rayon over rows with a per-thread score scratch (`for_each_init`);
    /// the per-timestep loops perform zero allocations.
    pub fn forward_events(&self, spikes: &SpikeTrain, ln2: &LayerNorm) -> Array2<f32> {
        let n = spikes.n;
        let t_steps = spikes.timesteps;
        let f = spikes.dim;
        debug_assert_eq!(self.query.weight.nrows(), f);
        let gamma = ln2.gamma.as_slice().unwrap();
        let beta = ln2.beta.as_slice().unwrap();
        let eps = ln2.eps;

        // q = query(mean over T): reconstruct the timestep mean sparsely.
        let mut mean = Array2::<f32>::zeros((n, f));
        mean.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(ni, mut row)| {
                let row = row.as_slice_mut().unwrap();
                let mut coef = 0.0_f32; // sum_t (-m_t * inv_std_t)
                for ti in 0..t_steps {
                    let mask = spikes.mask_row(ni, ti);
                    let k = mask.iter().map(|w| w.count_ones()).sum();
                    let (m, inv_std) = SpikeTrain::ln_scalars(k, f, eps);
                    coef -= m * inv_std;
                    for_each_set_bit(mask, |i| row[i] += inv_std * gamma[i]);
                }
                let inv_t = 1.0 / t_steps as f32;
                for (i, v) in row.iter_mut().enumerate() {
                    // (sparse + coef*gamma + T*beta) / T
                    *v = (*v + coef * gamma[i]) * inv_t + beta[i];
                }
            });
        let q = self.query.forward(&mean.view());

        // Fused ln2 -> key operands (precomputed once per call).
        let wk = &self.key.weight; // (f, f)
        let kg = Array2::from_shape_fn((f, f), |(i, o)| gamma[i] * wk[(i, o)]);
        let gw_k = ln2.gamma.dot(wk);
        let mut bw_k = ln2.beta.dot(wk);
        if let Some(b) = &self.key.bias {
            bw_k += b;
        }
        // qg[n,i] = dot(q_n, gamma_i * W_k[i,:]) — one (n,f)x(f,f) GEMM, so
        // each score needs only a popcount-driven sum of qg entries.
        let qg = q.dot(&kg.t());
        let c1 = q.dot(&gw_k); // dot(q_n, gamma^T W_k)       (len n)
        let c2 = q.dot(&bw_k); // dot(q_n, beta^T W_k + bias) (len n)
        let qg_s = qg.as_slice().unwrap();

        let scale = self.scale;
        let mut out = Array2::<f32>::zeros((n, f));
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each_init(
                || vec![0.0_f32; t_steps],
                |scores, (ni, mut orow)| {
                    let orow = orow.as_slice_mut().unwrap();
                    let qg_row = &qg_s[ni * f..(ni + 1) * f];
                    // Scores over T from set bits + rank-1 corrections.
                    for (ti, sc) in scores.iter_mut().enumerate() {
                        let mask = spikes.mask_row(ni, ti);
                        let k = mask.iter().map(|w| w.count_ones()).sum();
                        let (m, inv_std) = SpikeTrain::ln_scalars(k, f, eps);
                        let mut acc = 0.0_f32;
                        for_each_set_bit(mask, |i| acc += qg_row[i]);
                        *sc = (inv_std * acc - m * inv_std * c1[ni] + c2[ni]) / scale;
                    }
                    softmax_slice(scores);
                    // Weighted timestep sum, again sparse + rank-1.
                    let mut coef = 0.0_f32; // sum_t (-w_t * m_t * inv_std_t)
                    for (ti, &w) in scores.iter().enumerate() {
                        let mask = spikes.mask_row(ni, ti);
                        let k = mask.iter().map(|wd| wd.count_ones()).sum();
                        let (m, inv_std) = SpikeTrain::ln_scalars(k, f, eps);
                        coef -= w * m * inv_std;
                        let wi = w * inv_std;
                        for_each_set_bit(mask, |i| orow[i] += wi * gamma[i]);
                    }
                    // Softmax weights sum to 1, so beta contributes once.
                    for (i, v) in orow.iter_mut().enumerate() {
                        *v += coef * gamma[i] + beta[i];
                    }
                },
            );
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

    // -- event-driven / sparse paths -------------------------------------

    /// Materialise the post-`ln2` float cube from a bit-packed spike train
    /// using the binary-LayerNorm two-value identity (reference consumer).
    fn reconstruct_ln(train: &SpikeTrain, ln: &LayerNorm) -> Array3<f32> {
        let mut out = Array3::<f32>::zeros((train.n, train.timesteps, train.dim));
        for ni in 0..train.n {
            for ti in 0..train.timesteps {
                let mask = train.mask_row(ni, ti);
                let k = train.popcount_row(ni, ti);
                let (m, inv_std) = SpikeTrain::ln_scalars(k, train.dim, ln.eps);
                for i in 0..train.dim {
                    let s = (mask[i / 64] >> (i % 64) & 1) as f32;
                    out[(ni, ti, i)] = (s - m) * inv_std * ln.gamma[i] + ln.beta[i];
                }
            }
        }
        out
    }

    fn max_abs_diff<'a>(
        a: impl IntoIterator<Item = &'a f32>,
        b: impl IntoIterator<Item = &'a f32>,
    ) -> f32 {
        a.into_iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn assert_rel_close<'a>(
        a: impl IntoIterator<Item = &'a f32>,
        b: impl IntoIterator<Item = &'a f32>,
        tol: f32,
    ) {
        for (i, (x, y)) in a.into_iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= tol * y.abs().max(1.0),
                "element {i}: {x} vs {y} exceeds rel tol {tol}"
            );
        }
    }

    #[test]
    fn forward_events_matches_dense_forward() {
        let mut rng = seeded_rng(42);
        let layer = SnnLayer::new(12, 32, 16, 6, &mut rng);
        let x = det_input(8, 12) * 3.0;

        let dense = layer.forward(&x.view());
        let train = layer.forward_events(&x.view());
        assert_eq!((train.n, train.timesteps, train.dim), (8, 6, 16));
        assert!(
            train.masks.iter().any(|&w| w != 0),
            "drive must elicit spikes for the comparison to be meaningful"
        );

        // The dense forward output IS the ln2'd spike cube, so reconstructing
        // ln2 values from the bit-packed spikes must reproduce it. This also
        // proves the spike patterns match exactly: a single flipped bit would
        // shift an entry by inv_std >= 1/sqrt(0.25 + eps) ~ 2.0 >> 1e-4.
        let rec = reconstruct_ln(&train, &layer.ln2);
        let diff = max_abs_diff(dense.iter(), rec.iter());
        assert!(diff <= 1e-4, "max abs diff {diff} vs dense forward");
    }

    #[test]
    fn temporal_attention_events_matches_dense() {
        let mut rng = seeded_rng(42);
        let layer = SnnLayer::new(12, 32, 16, 6, &mut rng);
        let attn = TemporalAttention::new(16, &mut rng);
        let x = det_input(8, 12) * 3.0;

        let cube = layer.forward(&x.view());
        let dense = attn.forward(&cube.view());
        let train = layer.forward_events(&x.view());
        let events = attn.forward_events(&train, layer.ln2());

        assert_eq!(events.dim(), dense.dim());
        assert_rel_close(events.iter(), dense.iter(), 1e-3);
    }

    #[test]
    fn end_to_end_events_matches_dense_pipeline() {
        // No pruning: the full event-driven pipeline must match the dense one.
        let mut rng = seeded_rng(9);
        let layer = SnnLayer::new(10, 24, 16, 8, &mut rng);
        let attn = TemporalAttention::new(16, &mut rng);
        let x = det_input(6, 10) * 3.0;

        let dense = attn.forward(&layer.forward(&x.view()).view());
        let events = attn.forward_events(&layer.forward_events(&x.view()), layer.ln2());

        assert_eq!(events.dim(), dense.dim());
        assert_rel_close(events.iter(), dense.iter(), 1e-3);
    }

    #[test]
    fn prune_recurrent_full_keep_restores_dense() {
        let x = det_input(4, 12) * 3.0;
        let mut rng_a = seeded_rng(42);
        let mut rng_b = seeded_rng(42);
        let base = SnnLayer::new(12, 16, 8, 4, &mut rng_a);
        let mut pruned = SnnLayer::new(12, 16, 8, 4, &mut rng_b);

        // Prune down, then restore: keep == 1.0 must be bit-identical dense.
        pruned.prune_recurrent(0.5);
        pruned.prune_recurrent(1.0);
        assert_eq!(base.forward(&x.view()), pruned.forward(&x.view()));
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb).max(1e-12)
    }

    #[test]
    fn prune_recurrent_half_output_stays_close() {
        // Pruning is an approximation: sanity-bound it by cosine similarity.
        //
        // (1) The quantity pruning directly approximates is the recurrent
        // drive `mem @ W_rec^T`: keeping the top half of an orthogonal-init
        // matrix by magnitude retains ~93% of its energy, so the drive
        // cosine is ~0.96 (measured 0.957-0.967 across sizes 32-256).
        let mut rng = seeded_rng(42);
        let mut neurons = PlifNeuronLayer::new(128, true, &mut rng);
        neurons.prune_recurrent(0.5);
        let mem = det_input(8, 128);
        let dense_drive = mem.dot(neurons.w_rec_t.as_ref().unwrap());
        let sparse_drive = neurons.w_rec_csr.as_ref().unwrap().matvec(&mem);
        let drive_cos = cosine(
            dense_drive.as_slice().unwrap(),
            sparse_drive.as_slice().unwrap(),
        );
        assert!(drive_cos > 0.95, "recurrent drive cosine {drive_cos} too low");

        // (2) End-to-end SnnLayer output: spikes are discrete, so any drive
        // perturbation that flips a spike moves a whole post-LN level —
        // the binary output amplifies the error (measured ~0.84-0.90 at
        // keep = 0.5). Bound it loosely as a sanity check.
        let x = det_input(8, 12) * 3.0;
        let mut rng_a = seeded_rng(42);
        let mut rng_b = seeded_rng(42);
        let base = SnnLayer::new(12, 32, 16, 6, &mut rng_a);
        let mut pruned = SnnLayer::new(12, 32, 16, 6, &mut rng_b);
        pruned.prune_recurrent(0.5);

        let ya = base.forward(&x.view());
        let yb = pruned.forward(&x.view());
        assert_eq!(ya.dim(), yb.dim());
        let out_cos = cosine(ya.as_slice().unwrap(), yb.as_slice().unwrap());
        assert!(out_cos > 0.8, "output cosine {out_cos} too low after pruning");
    }

    #[test]
    fn prune_recurrent_deterministic() {
        let x = det_input(5, 12) * 3.0;
        let mut rng_a = seeded_rng(42);
        let mut rng_b = seeded_rng(42);
        let mut a = SnnLayer::new(12, 16, 8, 4, &mut rng_a);
        let mut b = SnnLayer::new(12, 16, 8, 4, &mut rng_b);
        a.prune_recurrent(0.3);
        b.prune_recurrent(0.3);
        let ya = a.forward(&x.view());
        assert_eq!(ya, b.forward(&x.view()));
        // Repeated runs of the same pruned layer are reproducible too.
        assert_eq!(ya, a.forward(&x.view()));
    }

    #[test]
    fn spike_train_memory_bytes_exact() {
        // dim = 128 -> 2 u64 words = 16 bytes = dim/8 per (n, t) row.
        let train = SpikeTrain {
            n: 8,
            timesteps: 6,
            dim: 128,
            words_per_row: 2,
            masks: vec![0; 8 * 6 * 2],
        };
        assert_eq!(train.memory_bytes(), 8 * 6 * (128 / 8));
    }

    #[test]
    fn spike_train_ln_scalars_edge_cases() {
        // k = 0: m = 0, var = 0 -> inv_std = 1/sqrt(eps), matching dense
        // LayerNorm on an all-zero row.
        let (m, inv_std) = SpikeTrain::ln_scalars(0, 128, 1e-5);
        assert_eq!(m, 0.0);
        assert!((inv_std - 1.0 / 1e-5_f32.sqrt()).abs() < 1e-2);
        // k = dim: var = 0 again (constant row).
        let (m, _) = SpikeTrain::ln_scalars(128, 128, 1e-5);
        assert_eq!(m, 1.0);
        // Half full: maximal variance 0.25.
        let (m, inv_std) = SpikeTrain::ln_scalars(64, 128, 1e-5);
        assert_eq!(m, 0.5);
        assert!((inv_std - 1.0 / (0.25_f32 + 1e-5).sqrt()).abs() < 1e-5);
    }
}
