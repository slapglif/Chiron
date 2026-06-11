//! HTM spatial pooler: Rust port of `chiron/layers/htm/model.py`.
//!
//! Implements the differentiable-HTM forward semantics on CPU:
//! potential-synapse masks per minicolumn, permanence-thresholded
//! connections, GEMM overlap, exponential boosting with EMA duty cycles,
//! global top-k inhibition, and the batched Hebbian permanence update.
//! Training-only autograd machinery (straight-through estimators, sigmoid
//! relaxation of the connected mask) is identity in forward value and is
//! therefore replaced by the exact hard threshold.
//!
//! Optimizations over the PyTorch reference:
//! - The effective connected matrix (`permanences >= syn_perm_connected`
//!   AND potential) is **cached transposed** as `(input_size, C)` so the
//!   overlap is a single GEMM `x.dot(&connected_t)`; the cache is rebuilt
//!   only when permanences change (Hebbian update / construction), never
//!   on a plain inference forward.
//! - Top-k inhibition uses an O(C) select-nth threshold
//!   (`nn::topk_threshold`) and emits **exactly** `num_active` ones per
//!   row (first-by-index among ties, like `torch.topk`).
//! - All per-row work is rayon-parallel; the Hebbian co-activation matrix
//!   is one GEMM (`active^T @ x`).
//! - [`HtmModel::forward`] pads/truncates the incoming features to
//!   `input_size` with a single zero-alloc + block copy (the hot path:
//!   64-dim GAT output into a 128-wide pooler).
//!
//! Deterministic given `HtmParams::seed` (potential masks via a seeded
//! partial Fisher-Yates shuffle; permanences via seeded uniform draws).

use ndarray::parallel::prelude::*;
use ndarray::{s, Array1, Array2, ArrayView2, Axis, Zip};
use rand::rngs::StdRng;
use rand::Rng;

use crate::nn::{seeded_rng, topk_threshold, LayerNorm, Linear};

// ---------------------------------------------------------------------------
// HtmParams
// ---------------------------------------------------------------------------

/// Hyper-parameters for the HTM spatial pooler.
///
/// `Default` matches `htm_params` in `chiron/config.json`.
#[derive(Clone, Debug)]
pub struct HtmParams {
    pub input_size: usize,
    pub minicolumn_size: usize,
    pub potential_radius: usize,
    pub potential_pct: f32,
    pub global_inhibition: bool,
    pub local_area_density: f32,
    pub num_active_columns_per_inhibition_area: usize,
    pub stimulus_threshold: f32,
    pub syn_perm_inactive_dec: f32,
    pub syn_perm_active_inc: f32,
    pub syn_perm_connected: f32,
    pub min_pct_overlap_duty_cycle: f32,
    pub duty_cycle_period: usize,
    pub max_boost: f32,
    pub seed: u64,
    pub sigmoid_sharpness: f32,
}

impl Default for HtmParams {
    fn default() -> Self {
        Self {
            input_size: 128,
            minicolumn_size: 16,
            potential_radius: 100,
            potential_pct: 0.5,
            global_inhibition: true,
            local_area_density: -1.0,
            num_active_columns_per_inhibition_area: 10,
            stimulus_threshold: 0.0,
            syn_perm_inactive_dec: 0.01,
            syn_perm_active_inc: 0.1,
            syn_perm_connected: 0.2,
            min_pct_overlap_duty_cycle: 0.005,
            duty_cycle_period: 500,
            max_boost: 5.0,
            seed: 42,
            sigmoid_sharpness: 10.0,
        }
    }
}

// ---------------------------------------------------------------------------
// HtmSpatialPooler
// ---------------------------------------------------------------------------

/// HTM spatial pooler with global inhibition, boosting and Hebbian learning.
pub struct HtmSpatialPooler {
    pub input_size: usize,
    pub num_minicolumns: usize,
    pub num_active: usize,
    pub syn_perm_connected: f32,
    pub syn_perm_active_inc: f32,
    pub syn_perm_inactive_dec: f32,
    pub max_boost: f32,
    ema_alpha: f32,
    /// Binary potential-synapse mask, `(C, input_size)`.
    pub potential_mask: Array2<f32>,
    /// Synapse permanences in `[0, 1]`, `(C, input_size)`.
    pub permanences: Array2<f32>,
    pub active_duty_cycles: Array1<f32>,
    pub overlap_duty_cycles: Array1<f32>,
    pub boosting_factors: Array1<f32>,
    pub step_count: u64,
    /// Cached connected matrix, stored transposed `(input_size, C)` so the
    /// overlap is one GEMM. Rebuilt only when permanences change.
    connected_t: Array2<f32>,
}

impl HtmSpatialPooler {
    pub fn new(params: &HtmParams) -> Self {
        let input_size = params.input_size;
        let minicolumn_size = params.minicolumn_size;
        assert!(input_size > 0, "HtmSpatialPooler: input_size must be > 0");
        assert!(
            minicolumn_size > 0,
            "HtmSpatialPooler: minicolumn_size must be > 0"
        );
        let num_minicolumns = input_size.div_ceil(minicolumn_size);
        let num_active = params
            .num_active_columns_per_inhibition_area
            .min(num_minicolumns);
        assert!(num_active >= 1, "HtmSpatialPooler: num_active must be >= 1");
        let ema_alpha = 2.0 / (params.duty_cycle_period as f32 + 1.0);

        let mut rng = seeded_rng(params.seed);

        // Potential mask: each column samples potential_pct of the inputs
        // within potential_radius of its centre (seeded partial Fisher-Yates).
        let mut potential_mask = Array2::<f32>::zeros((num_minicolumns, input_size));
        let half_r = params.potential_radius / 2;
        for col in 0..num_minicolumns {
            let centre = col * input_size / num_minicolumns;
            let lo = centre.saturating_sub(half_r);
            let hi = (centre + half_r + 1).min(input_size);
            let span = hi - lo;
            let n_potential = ((span as f32 * params.potential_pct) as usize)
                .max(1)
                .min(span);
            let mut idx: Vec<usize> = (lo..hi).collect();
            for i in 0..n_potential {
                let j = rng.gen_range(i..span);
                idx.swap(i, j);
            }
            for &k in &idx[..n_potential] {
                potential_mask[[col, k]] = 1.0;
            }
        }

        // Permanences around the connected threshold, masked and clamped.
        let (p_lo, p_hi) = (
            params.syn_perm_connected - 0.05,
            params.syn_perm_connected + 0.05,
        );
        let mut permanences =
            Array2::from_shape_fn((num_minicolumns, input_size), |_| rng.gen_range(p_lo..p_hi));
        Zip::from(&mut permanences)
            .and(&potential_mask)
            .for_each(|p, &m| *p = (*p * m).clamp(0.0, 1.0));

        let mut pooler = Self {
            input_size,
            num_minicolumns,
            num_active,
            syn_perm_connected: params.syn_perm_connected,
            syn_perm_active_inc: params.syn_perm_active_inc,
            syn_perm_inactive_dec: params.syn_perm_inactive_dec,
            max_boost: params.max_boost,
            ema_alpha,
            potential_mask,
            permanences,
            active_duty_cycles: Array1::zeros(num_minicolumns),
            overlap_duty_cycles: Array1::zeros(num_minicolumns),
            boosting_factors: Array1::ones(num_minicolumns),
            step_count: 0,
            connected_t: Array2::zeros((input_size, num_minicolumns)),
        };
        pooler.rebuild_connected();
        pooler
    }

    /// Rebuild the cached transposed connected matrix from permanences.
    fn rebuild_connected(&mut self) {
        let thr = self.syn_perm_connected;
        let mut ct = Array2::<f32>::zeros((self.input_size, self.num_minicolumns));
        for (c, (perm_row, mask_row)) in self
            .permanences
            .rows()
            .into_iter()
            .zip(self.potential_mask.rows())
            .enumerate()
        {
            for (i, (&p, &m)) in perm_row.iter().zip(mask_row).enumerate() {
                if p >= thr && m > 0.0 {
                    ct[[i, c]] = 1.0;
                }
            }
        }
        self.connected_t = ct;
    }

    /// Effective binary connected mask `(C, input_size)` (forward value of
    /// the Python sigmoid/STE construction): `(perm >= thr) * potential`.
    pub fn connected_mask(&self) -> Array2<f32> {
        self.connected_t.t().as_standard_layout().to_owned()
    }

    /// Global inhibition: boost overlaps, then exactly `num_active` ones
    /// per row (first-by-index among threshold ties, like `torch.topk`).
    fn select_active(&self, overlap: &Array2<f32>) -> Array2<f32> {
        let k = self.num_active;
        let boost = self
            .boosting_factors
            .as_slice()
            .expect("boosting_factors is contiguous");
        let mut mask = overlap.clone();
        mask.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut row| {
            let r = row.as_slice_mut().expect("row is contiguous");
            for (v, b) in r.iter_mut().zip(boost) {
                *v *= b;
            }
            let thr = topk_threshold(r, k);
            let mut remaining = k;
            for v in r.iter_mut() {
                if remaining > 0 && *v >= thr {
                    *v = 1.0;
                    remaining -= 1;
                } else {
                    *v = 0.0;
                }
            }
        });
        mask
    }

    /// EMA duty-cycle update and exponential boosting (training only).
    fn update_homeostasis(&mut self, overlap: &Array2<f32>, active: &Array2<f32>) {
        let alpha = self.ema_alpha;
        let n = overlap.nrows() as f32;
        let c = self.num_minicolumns;

        let batch_active_rate = active.sum_axis(Axis(0)) / n;
        let mut batch_overlap_rate = Array1::<f32>::zeros(c);
        for row in overlap.rows() {
            for (acc, &v) in batch_overlap_rate.iter_mut().zip(row) {
                if v > 0.0 {
                    *acc += 1.0;
                }
            }
        }
        batch_overlap_rate /= n;

        self.active_duty_cycles =
            &self.active_duty_cycles * (1.0 - alpha) + &(batch_active_rate * alpha);
        self.overlap_duty_cycles =
            &self.overlap_duty_cycles * (1.0 - alpha) + &(batch_overlap_rate * alpha);
        self.step_count += 1;

        let target_density = (self.active_duty_cycles.sum() / c as f32).max(1e-6);
        let max_boost = self.max_boost;
        self.boosting_factors = self
            .active_duty_cycles
            .mapv(|d| (max_boost * (target_density - d)).exp().clamp(1.0, max_boost));
    }

    /// Batched Hebbian permanence update (training only). One GEMM for the
    /// co-activation counts, then a fused rayon-parallel add+clamp per row.
    fn hebbian_update(&mut self, active: &Array2<f32>, x: &ArrayView2<f32>) {
        let n = x.nrows() as f32;
        let coactive = active.t().dot(x) / n; // (C, input_size)
        let active_count = active.sum_axis(Axis(0)) / n; // (C,)

        let inc = self.syn_perm_active_inc;
        let dec = self.syn_perm_inactive_dec;
        Zip::from(self.permanences.rows_mut())
            .and(coactive.rows())
            .and(self.potential_mask.rows())
            .and(&active_count)
            .par_for_each(|mut perm_row, co_row, mask_row, &acnt| {
                for ((p, &co), &m) in perm_row.iter_mut().zip(co_row).zip(mask_row) {
                    let delta = (inc * co - dec * (acnt - co)) * m;
                    *p = (*p + delta).clamp(0.0, 1.0);
                }
            });

        self.rebuild_connected();
    }

    /// Run the pooler on a flat batch `x: (N, input_size)`. Returns the
    /// binary active-column mask `(N, C)` with exactly `num_active` ones
    /// per row. With `training`, also performs the homeostatic duty-cycle /
    /// boosting update and the Hebbian permanence update.
    pub fn forward(&mut self, x: &ArrayView2<f32>, training: bool) -> Array2<f32> {
        assert_eq!(
            x.ncols(),
            self.input_size,
            "HtmSpatialPooler: expected input width {}, got {}",
            self.input_size,
            x.ncols()
        );
        let overlap = x.dot(&self.connected_t); // (N, C) — single GEMM
        let active = self.select_active(&overlap);
        if training && overlap.nrows() > 0 {
            self.update_homeostasis(&overlap, &active);
            self.hebbian_update(&active, x);
        }
        active
    }
}

// ---------------------------------------------------------------------------
// HtmModel
// ---------------------------------------------------------------------------

/// High-level wrapper: pad/project the input to the pooler width, run the
/// spatial pooler, project the column activations to `sdr_dimensions`, add
/// a residual path, and LayerNorm.
pub struct HtmModel {
    pub spatial_pooler: HtmSpatialPooler,
    pub sdr_dimensions: usize,
    input_proj: Linear,
    fc: Linear,
    residual_proj: Option<Linear>,
    layer_norm: LayerNorm,
}

impl HtmModel {
    pub fn new(sdr_dimensions: usize, params: &HtmParams, rng: &mut StdRng) -> Self {
        assert!(sdr_dimensions > 0, "HtmModel: sdr_dimensions must be > 0");
        let spatial_pooler = HtmSpatialPooler::new(params);
        let input_size = spatial_pooler.input_size;
        let num_cols = spatial_pooler.num_minicolumns;

        let input_proj = Linear::new(input_size, input_size, true, rng);
        let fc = Linear::new(num_cols, sdr_dimensions, true, rng);
        let residual_proj = (input_size != sdr_dimensions)
            .then(|| Linear::new(input_size, sdr_dimensions, true, rng));

        Self {
            spatial_pooler,
            sdr_dimensions,
            input_proj,
            fc,
            residual_proj,
            layer_norm: LayerNorm::new(sdr_dimensions),
        }
    }

    /// `input: (N, feat) -> (N, sdr_dimensions)` where `N = batch * seq`.
    ///
    /// `feat` is right-truncated or zero-padded to the pooler's
    /// `input_size` (hot path: 64-dim GAT features into a 128-wide pooler;
    /// one zeroed allocation plus a block copy).
    pub fn forward(&mut self, input: &ArrayView2<f32>, training: bool) -> Array2<f32> {
        let n = input.nrows();
        let feat = input.ncols();
        let input_size = self.spatial_pooler.input_size;

        let padded_owned: Array2<f32>;
        let padded: ArrayView2<f32> = if feat == input_size {
            input.view()
        } else {
            let mut z = Array2::<f32>::zeros((n, input_size));
            let w = feat.min(input_size);
            z.slice_mut(s![.., ..w]).assign(&input.slice(s![.., ..w]));
            padded_owned = z;
            padded_owned.view()
        };

        let x = self.input_proj.forward(&padded);
        let sp_out = self.spatial_pooler.forward(&x.view(), training);
        let mut out = self.fc.forward(&sp_out.view());
        match &self.residual_proj {
            Some(proj) => out += &proj.forward(&padded),
            None => out += &padded,
        }
        self.layer_norm.forward_inplace(out.view_mut());
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn random_input(n: usize, d: usize, seed: u64) -> Array2<f32> {
        let mut rng = seeded_rng(seed);
        Array2::from_shape_fn((n, d), |_| rng.gen::<f32>())
    }

    #[test]
    fn default_params_exactly_num_active_per_row() {
        // C = ceil(128/16) = 8, num_active = min(10, 8) = 8.
        let params = HtmParams::default();
        let mut pooler = HtmSpatialPooler::new(&params);
        assert_eq!(pooler.num_minicolumns, 8);
        assert_eq!(pooler.num_active, 8);
        let x = random_input(12, 128, 1);
        let active = pooler.forward(&x.view(), false);
        assert_eq!(active.shape(), &[12, 8]);
        for row in active.rows() {
            assert_eq!(row.sum(), 8.0, "expected exactly num_active ones");
        }
    }

    #[test]
    fn fewer_active_columns_exact_topk_even_with_ties() {
        let params = HtmParams {
            num_active_columns_per_inhibition_area: 3,
            ..Default::default()
        };
        let mut pooler = HtmSpatialPooler::new(&params);
        assert_eq!(pooler.num_active, 3);
        // Binary input produces integer overlaps -> threshold ties are
        // likely; output must still have exactly 3 ones per row.
        let x = random_input(20, 128, 2).mapv(|v| if v > 0.5 { 1.0 } else { 0.0 });
        let active = pooler.forward(&x.view(), false);
        for row in active.rows() {
            assert_eq!(row.sum(), 3.0, "exactly k ones per row, like torch.topk");
            assert!(row.iter().all(|&v| v == 0.0 || v == 1.0));
        }
    }

    #[test]
    fn htm_model_pads_64_to_128_and_outputs_sdr_dims() {
        let params = HtmParams::default();
        let mut rng = seeded_rng(0);
        let mut model = HtmModel::new(256, &params, &mut rng);
        let x = random_input(6, 64, 3); // hot path: pad 64 -> 128
        let out = model.forward(&x.view(), false);
        assert_eq!(out.shape(), &[6, 256]);
        // Truncation path (feat > input_size) also produces the right shape.
        let wide = random_input(4, 200, 4);
        let out = model.forward(&wide.view(), false);
        assert_eq!(out.shape(), &[4, 256]);
    }

    #[test]
    fn pooler_and_model_are_deterministic() {
        let params = HtmParams::default();
        let x = random_input(10, 128, 5);

        let mut p1 = HtmSpatialPooler::new(&params);
        let mut p2 = HtmSpatialPooler::new(&params);
        assert_eq!(p1.permanences, p2.permanences);
        assert_eq!(p1.potential_mask, p2.potential_mask);
        for _ in 0..3 {
            let a = p1.forward(&x.view(), true);
            let b = p2.forward(&x.view(), true);
            assert_eq!(a, b);
        }
        assert_eq!(p1.permanences, p2.permanences);
        assert_eq!(p1.boosting_factors, p2.boosting_factors);

        let e = random_input(5, 64, 6);
        let mut m1 = HtmModel::new(256, &params, &mut seeded_rng(7));
        let mut m2 = HtmModel::new(256, &params, &mut seeded_rng(7));
        assert_eq!(m1.forward(&e.view(), false), m2.forward(&e.view(), false));
    }

    #[test]
    fn training_keeps_state_in_valid_ranges() {
        let params = HtmParams::default();
        let mut pooler = HtmSpatialPooler::new(&params);
        for step in 0..10 {
            let x = random_input(16, 128, 100 + step);
            pooler.forward(&x.view(), true);
        }
        assert_eq!(pooler.step_count, 10);
        for &b in pooler.boosting_factors.iter() {
            assert!((1.0..=params.max_boost).contains(&b), "boost {b} out of range");
        }
        for &d in pooler
            .active_duty_cycles
            .iter()
            .chain(pooler.overlap_duty_cycles.iter())
        {
            assert!((0.0..=1.0).contains(&d), "duty cycle {d} out of range");
        }
        for &p in pooler.permanences.iter() {
            assert!((0.0..=1.0).contains(&p), "permanence {p} out of range");
        }
    }

    #[test]
    fn hebbian_update_strengthens_coactive_pairs_and_boosts_starved_columns() {
        // Tiny pooler: input_size=8, minicolumn_size=4 -> C=2; full potential
        // coverage (radius 100, pct 1.0); one winner per row.
        let params = HtmParams {
            input_size: 8,
            minicolumn_size: 4,
            potential_pct: 1.0,
            num_active_columns_per_inhibition_area: 1,
            ..Default::default()
        };
        let mut pooler = HtmSpatialPooler::new(&params);
        assert_eq!(pooler.num_minicolumns, 2);
        assert!(pooler.potential_mask.iter().all(|&m| m == 1.0));

        let perms_before = pooler.permanences.clone();
        // Active inputs 0..4, silent inputs 4..8.
        let mut x = Array2::<f32>::zeros((1, 8));
        x.slice_mut(s![0, ..4]).fill(1.0);
        let active = pooler.forward(&x.view(), true);
        assert_eq!(active.row(0).sum(), 1.0);
        let winner = if active[[0, 0]] == 1.0 { 0 } else { 1 };
        let loser = 1 - winner;

        for i in 0..8 {
            let before = perms_before[[winner, i]];
            let after = pooler.permanences[[winner, i]];
            if i < 4 {
                // Co-active (column, input) pair: +active_inc.
                assert!(
                    (after - (before + params.syn_perm_active_inc)).abs() < 1e-6,
                    "co-active synapse {i} not strengthened: {before} -> {after}"
                );
            } else {
                // Active column, silent input: -inactive_dec.
                assert!(
                    (after - (before - params.syn_perm_inactive_dec)).abs() < 1e-6,
                    "silent-input synapse {i} not weakened: {before} -> {after}"
                );
            }
            // Inactive column: untouched.
            assert_eq!(pooler.permanences[[loser, i]], perms_before[[loser, i]]);
        }

        // Homeostasis: the starved column gets a boost > 1, the winner is
        // clamped at 1.
        assert!(pooler.boosting_factors[loser] > 1.0, "starved column not boosted");
        assert_eq!(pooler.boosting_factors[winner], 1.0);
    }
}
