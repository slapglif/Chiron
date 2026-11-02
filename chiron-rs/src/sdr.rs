//! SDR generation: Rust port of `chiron/layers/sdr/sdr_generation.py`.
//!
//! Pipeline: lazily-built input projector -> LayerNorm -> FlyHash sparse
//! random expansion (Dasgupta et al., 2017) -> k-Winners-Take-All with
//! homeostatic boosting. Output is a binary `(N, sdr_dimensions)` SDR with
//! exactly `max(1, round(n * sparsity))` active bits per row (modulo ties).
//!
//! Optimizations over the PyTorch reference:
//! - The FlyHash projection matrix is stored **pre-transposed** as
//!   `(input_dim, expansion_dim)` so the forward pass is a single GEMM
//!   `x.dot(&w)` with no per-call transpose.
//! - k-WTA thresholds use an O(n) select-nth (`nn::topk_threshold`)
//!   instead of a full sort, and the winner mask is written in place.
//! - All per-row work (gain scaling, boosting, thresholding, bit packing)
//!   is rayon-parallel across the batch.
//! - [`BitSdr`] packs binary SDRs 64 bits/word so overlap / Jaccard /
//!   Hamming similarity reduce to a handful of `AND`/`XOR` + popcount ops,
//!   versus the dense float arrays the Python pipeline carries around.
//!
//! Everything is deterministic given the construction seed. Training-only
//! stochastic parts of the reference (straight-through estimators, dropout)
//! are identity in forward value and therefore omitted.

use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::nn::{seeded_rng, topk_threshold, LayerNorm, Linear};

// ---------------------------------------------------------------------------
// FlyHashProjection
// ---------------------------------------------------------------------------

/// Sparse random projection inspired by the Drosophila olfactory circuit.
///
/// Each expansion neuron connects to a random ~`min(10/input_dim, 0.5)`
/// fraction of inputs with Gaussian weights; each neuron's weight vector is
/// L2-normalized. A learnable per-neuron gain modulates the fixed projection.
pub struct FlyHashProjection {
    /// Projection weights stored transposed: `(input_dim, expansion_dim)`.
    /// Column `j` is the (L2-normalized) weight vector of expansion neuron `j`.
    pub weight_t: Array2<f32>,
    /// Learnable gain, one per expansion neuron.
    pub gain: Array1<f32>,
}

impl FlyHashProjection {
    pub fn new(input_dim: usize, expansion_dim: usize, seed: u64) -> Self {
        assert!(input_dim > 0, "FlyHashProjection: input_dim must be > 0");
        assert!(
            expansion_dim > 0,
            "FlyHashProjection: expansion_dim must be > 0"
        );
        let mut rng = seeded_rng(seed);
        let connection_prob = (10.0 / input_dim as f32).min(0.5);
        let normal = Normal::new(0.0_f32, 1.0).expect("valid normal params");

        // Build row-major (expansion_dim, input_dim) sequentially for
        // deterministic RNG draw order, normalizing each neuron's row.
        let mut w = Array2::<f32>::zeros((expansion_dim, input_dim));
        for mut row in w.axis_iter_mut(Axis(0)) {
            let mut norm_sq = 0.0_f32;
            for v in row.iter_mut() {
                if rng.gen::<f32>() < connection_prob {
                    let s = normal.sample(&mut rng);
                    *v = s;
                    norm_sq += s * s;
                }
            }
            let inv = 1.0 / norm_sq.sqrt().max(1e-8);
            for v in row.iter_mut() {
                *v *= inv;
            }
        }

        // Store transposed (standard layout) so forward is one GEMM.
        let weight_t = w.t().as_standard_layout().to_owned();
        Self {
            weight_t,
            gain: Array1::ones(expansion_dim),
        }
    }

    /// `x: (N, input_dim) -> (N, expansion_dim)`: `x @ W^T` scaled by gain.
    pub fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let mut out = x.dot(&self.weight_t);
        let gain = self.gain.as_slice().expect("gain is contiguous");
        out.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut row| {
            let r = row.as_slice_mut().expect("row is contiguous");
            for (v, g) in r.iter_mut().zip(gain) {
                *v *= g;
            }
        });
        out
    }
}

// ---------------------------------------------------------------------------
// KWinnersTakeAll
// ---------------------------------------------------------------------------

/// k-Winners-Take-All competitive inhibition with duty-cycle boosting.
///
/// At inference returns the exact binary winner mask (the Python
/// straight-through `x + (winners - x).detach()` equals `winners` in
/// forward value). During training, columns are boosted by
/// `exp(boost_strength * (sparsity - duty_cycle))` before the competition
/// and duty cycles are EMA-updated from the batch winner rate.
pub struct KWinnersTakeAll {
    pub sparsity: f32,
    pub boost_strength: f32,
    duty_cycles: Option<Array1<f32>>,
}

impl KWinnersTakeAll {
    pub fn new(sparsity: f32, boost_strength: f32) -> Self {
        Self {
            sparsity,
            boost_strength,
            duty_cycles: None,
        }
    }

    /// Number of winners for an `n`-wide row.
    #[inline]
    pub fn compute_k(&self, n: usize) -> usize {
        ((n as f32 * self.sparsity).round() as usize).max(1)
    }

    /// Sparsify each row of `x` to its top-k entries (1.0 winners, 0.0 rest).
    ///
    /// Ties at the threshold are all kept, so a row may have more than `k`
    /// winners (matches the `x >= kth_value` semantics of the reference).
    pub fn forward(&mut self, mut x: Array2<f32>, training: bool) -> Array2<f32> {
        let (rows, n) = (x.nrows(), x.ncols());
        assert!(n > 0, "KWinnersTakeAll: input must have at least one column");
        let k = self.compute_k(n);

        // Homeostatic boosting (training only, once duty cycles exist).
        if training {
            if let Some(dc) = &self.duty_cycles {
                assert_eq!(dc.len(), n, "KWinnersTakeAll: duty-cycle width mismatch");
                let (bs, sp) = (self.boost_strength, self.sparsity);
                let boost: Vec<f32> = dc.iter().map(|&d| (bs * (sp - d)).exp()).collect();
                x.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut row| {
                    let r = row.as_slice_mut().expect("row is contiguous");
                    for (v, b) in r.iter_mut().zip(&boost) {
                        *v *= b;
                    }
                });
            }
        }

        // Winner-take-all: O(n) k-th largest per row, then in-place binarize.
        x.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut row| {
            let r = row.as_slice_mut().expect("row is contiguous");
            let thr = topk_threshold(r, k);
            for v in r.iter_mut() {
                *v = if *v >= thr { 1.0 } else { 0.0 };
            }
        });

        // Duty-cycle EMA from the batch winner rate.
        if training && rows > 0 {
            let batch_duty = x.sum_axis(Axis(0)) / rows as f32;
            self.duty_cycles = Some(match self.duty_cycles.take() {
                Some(old) => old * 0.99 + &(batch_duty * 0.01),
                None => batch_duty,
            });
        }

        x
    }
}

// ---------------------------------------------------------------------------
// SdrGenerator
// ---------------------------------------------------------------------------

/// End-to-end SDR generator: input projection -> LayerNorm -> FlyHash
/// expansion -> k-WTA sparsification.
pub struct SdrGenerator {
    pub projection_dimensions: usize,
    pub sdr_dimensions: usize,
    input_norm: LayerNorm,
    fly_hash: FlyHashProjection,
    kwta: KWinnersTakeAll,
    /// Lazily-created `(input_dim, Linear)` projector, rebuilt if the
    /// incoming feature width changes (mirrors `_get_input_projector`).
    input_projector: Option<(usize, Linear)>,
    rng: StdRng,
}

impl SdrGenerator {
    pub fn new(
        projection_dimensions: usize,
        sdr_dimensions: usize,
        sparsity: f32,
        seed: u64,
    ) -> Self {
        assert!(
            projection_dimensions < sdr_dimensions,
            "SdrGenerator: projection_dimensions ({projection_dimensions}) must be \
             less than sdr_dimensions ({sdr_dimensions})"
        );
        Self {
            projection_dimensions,
            sdr_dimensions,
            input_norm: LayerNorm::new(projection_dimensions),
            fly_hash: FlyHashProjection::new(projection_dimensions, sdr_dimensions, seed),
            kwta: KWinnersTakeAll::new(sparsity, 1.5),
            input_projector: None,
            rng: seeded_rng(seed),
        }
    }

    /// `embeddings: (N, D) -> (N, sdr_dimensions)` binary SDR batch.
    pub fn forward(&mut self, embeddings: &ArrayView2<f32>, training: bool) -> Array2<f32> {
        let input_dim = embeddings.ncols();
        assert!(input_dim > 0, "SdrGenerator: embeddings must have width > 0");

        let stale = !matches!(&self.input_projector, Some((d, _)) if *d == input_dim);
        if stale {
            let std = 1.0 / (input_dim as f32).sqrt();
            let lin = Linear::new_normal(input_dim, self.projection_dimensions, std, &mut self.rng);
            self.input_projector = Some((input_dim, lin));
        }
        let projector = &self.input_projector.as_ref().expect("just initialized").1;

        let mut projected = projector.forward(embeddings);
        self.input_norm.forward_inplace(projected.view_mut());
        let expanded = self.fly_hash.forward(&projected.view());
        self.kwta.forward(expanded, training)
    }
}

// ---------------------------------------------------------------------------
// Bit-packed SDRs
// ---------------------------------------------------------------------------

/// A binary SDR packed 64 bits per `u64` word. Similarity ops are
/// word-wise `AND`/`XOR`/`OR` + popcount: ~dim/64 instructions instead of
/// dim float multiply-adds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitSdr {
    pub dim: usize,
    pub words: Vec<u64>,
}

impl BitSdr {
    /// Pack one dense row; bit `i` is set where `row[i] > 0.5`.
    pub fn from_dense_row(row: &[f32]) -> Self {
        let dim = row.len();
        let mut words = vec![0u64; dim.div_ceil(64)];
        for (i, &v) in row.iter().enumerate() {
            if v > 0.5 {
                words[i >> 6] |= 1u64 << (i & 63);
            }
        }
        Self { dim, words }
    }

    /// Number of shared active bits: `popcount(self AND other)`.
    #[inline]
    pub fn overlap(&self, other: &BitSdr) -> u32 {
        debug_assert_eq!(self.dim, other.dim, "BitSdr dim mismatch");
        self.words
            .iter()
            .zip(&other.words)
            .map(|(a, b)| (a & b).count_ones())
            .sum()
    }

    /// Jaccard similarity `|AND| / |OR|`; 0.0 when both SDRs are empty.
    #[inline]
    pub fn jaccard(&self, other: &BitSdr) -> f32 {
        debug_assert_eq!(self.dim, other.dim, "BitSdr dim mismatch");
        let mut and = 0u32;
        let mut or = 0u32;
        for (a, b) in self.words.iter().zip(&other.words) {
            and += (a & b).count_ones();
            or += (a | b).count_ones();
        }
        if or == 0 {
            0.0
        } else {
            and as f32 / or as f32
        }
    }

    /// Hamming distance: `popcount(self XOR other)`.
    #[inline]
    pub fn hamming(&self, other: &BitSdr) -> u32 {
        debug_assert_eq!(self.dim, other.dim, "BitSdr dim mismatch");
        self.words
            .iter()
            .zip(&other.words)
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Total number of active bits.
    #[inline]
    pub fn active_bits(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }
}

/// Pack every row of a dense binary batch into [`BitSdr`]s (rayon-parallel).
pub fn pack_batch(dense: &ArrayView2<f32>) -> Vec<BitSdr> {
    dense
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| match row.as_slice() {
            Some(s) => BitSdr::from_dense_row(s),
            None => BitSdr::from_dense_row(&row.to_vec()),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn random_input(n: usize, d: usize, seed: u64) -> Array2<f32> {
        let mut rng = seeded_rng(seed);
        Array2::from_shape_fn((n, d), |_| rng.gen::<f32>() * 2.0 - 1.0)
    }

    #[test]
    fn sdr_exact_sparsity_one_bit() {
        // Config dims: 64 -> 256 at sparsity 0.002 => k = max(1, round(0.512)) = 1.
        let mut generator = SdrGenerator::new(64, 256, 0.002, 42);
        let x = random_input(8, 300, 1);
        let sdr = generator.forward(&x.view(), false);
        assert_eq!(sdr.shape(), &[8, 256]);
        for row in sdr.rows() {
            let active: f32 = row.sum();
            assert_eq!(active, 1.0, "expected exactly 1 active bit per row");
            assert!(row.iter().all(|&v| v == 0.0 || v == 1.0), "output not binary");
        }
    }

    #[test]
    fn sdr_exact_sparsity_five_bits() {
        // 256 dims at sparsity 0.02 => k = round(5.12) = 5.
        let mut generator = SdrGenerator::new(64, 256, 0.02, 7);
        let x = random_input(16, 128, 2);
        let sdr = generator.forward(&x.view(), false);
        for row in sdr.rows() {
            assert_eq!(row.sum(), 5.0, "expected exactly 5 active bits per row");
        }
    }

    #[test]
    fn sdr_generator_deterministic() {
        let x = random_input(10, 300, 3);
        let mut g1 = SdrGenerator::new(64, 256, 0.02, 42);
        let mut g2 = SdrGenerator::new(64, 256, 0.02, 42);
        let a = g1.forward(&x.view(), false);
        let b = g2.forward(&x.view(), false);
        assert_eq!(a, b, "same seed must produce identical SDRs");
        // Training path is deterministic too.
        let a = g1.forward(&x.view(), true);
        let b = g2.forward(&x.view(), true);
        assert_eq!(a, b);
    }

    #[test]
    fn sdr_generator_panics_on_bad_dims() {
        let result = std::panic::catch_unwind(|| SdrGenerator::new(256, 256, 0.02, 0));
        assert!(result.is_err(), "projection >= sdr dims must panic");
    }

    #[test]
    fn flyhash_rows_are_unit_norm() {
        let fh = FlyHashProjection::new(64, 256, 42);
        assert_eq!(fh.weight_t.shape(), &[64, 256]);
        // Each expansion neuron = one column of the transposed storage.
        for j in 0..256 {
            let norm: f32 = fh
                .weight_t
                .column(j)
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            // Rows with no sampled connections stay all-zero (norm clamp).
            assert!(
                (norm - 1.0).abs() < 1e-4 || norm < 1e-6,
                "neuron {j} norm {norm} is neither ~1 nor 0"
            );
        }
    }

    #[test]
    fn kwta_keeps_all_tied_winners() {
        let mut kwta = KWinnersTakeAll::new(0.5, 1.5);
        // n = 4, k = 2, but three values tie at the threshold.
        let x = array![[1.0_f32, 1.0, 1.0, 0.2]];
        let out = kwta.forward(x, false);
        let winners: f32 = out.row(0).sum();
        assert!(winners >= 2.0, "ties at the threshold must all win");
        assert_eq!(winners, 3.0);
        assert_eq!(out, array![[1.0_f32, 1.0, 1.0, 0.0]]);
    }

    #[test]
    fn kwta_training_updates_duty_cycles() {
        let mut kwta = KWinnersTakeAll::new(0.25, 1.5);
        let x = array![[0.9_f32, 0.1, 0.2, 0.3], [0.8, 0.0, 0.1, 0.2]];
        let out = kwta.forward(x.clone(), true);
        // k = 1; column 0 wins in both rows -> first-call duty = batch mean.
        assert_eq!(out.column(0).sum(), 2.0);
        let dc = kwta.duty_cycles.as_ref().expect("initialized on first call");
        assert_eq!(dc[0], 1.0);
        assert_eq!(dc[1], 0.0);
        // Second call: boosting suppresses the over-active column 0
        // (boost exp(1.5*(0.25-1)) ~ 0.32) so column 3 wins both rows;
        // duty cycles follow the EMA 0.99*old + 0.01*batch.
        let out = kwta.forward(x, true);
        assert_eq!(out.column(3).sum(), 2.0, "starved column should win after boost");
        let dc = kwta.duty_cycles.as_ref().unwrap();
        assert!((dc[0] - 0.99).abs() < 1e-6);
        assert!((dc[3] - 0.01).abs() < 1e-6);
    }

    #[test]
    fn bitsdr_small_hand_computed() {
        // dim = 5 (not a multiple of 64).
        let a = BitSdr::from_dense_row(&[1.0, 0.0, 1.0, 1.0, 0.0]); // {0, 2, 3}
        let b = BitSdr::from_dense_row(&[1.0, 1.0, 0.0, 1.0, 0.0]); // {0, 1, 3}
        assert_eq!(a.active_bits(), 3);
        assert_eq!(b.active_bits(), 3);
        assert_eq!(a.overlap(&b), 2); // {0, 3}
        assert_eq!(a.hamming(&b), 2); // {1, 2}
        let jac = a.jaccard(&b); // |{0,3}| / |{0,1,2,3}| = 2/4
        assert!((jac - 0.5).abs() < 1e-6);

        // Empty vs empty.
        let e1 = BitSdr::from_dense_row(&[0.0; 5]);
        let e2 = BitSdr::from_dense_row(&[0.0; 5]);
        assert_eq!(e1.jaccard(&e2), 0.0);
        assert_eq!(e1.overlap(&e2), 0);
        assert_eq!(e1.hamming(&e2), 0);
    }

    #[test]
    fn bitsdr_dim_not_multiple_of_64() {
        // dim = 70: bits straddle the word boundary.
        let mut da = vec![0.0_f32; 70];
        let mut db = vec![0.0_f32; 70];
        da[63] = 1.0;
        da[64] = 1.0;
        da[69] = 1.0;
        db[64] = 1.0;
        db[0] = 1.0;
        let a = BitSdr::from_dense_row(&da); // {63, 64, 69}
        let b = BitSdr::from_dense_row(&db); // {0, 64}
        assert_eq!(a.words.len(), 2);
        assert_eq!(a.active_bits(), 3);
        assert_eq!(a.overlap(&b), 1); // {64}
        assert_eq!(a.hamming(&b), 3); // {0, 63, 69}
        assert!((a.jaccard(&b) - 0.25).abs() < 1e-6); // 1 / |{0,63,64,69}|
    }

    #[test]
    fn pack_batch_matches_per_row_packing() {
        let dense = array![[1.0_f32, 0.0, 0.6, 0.4], [0.0, 0.51, 0.0, 1.0]];
        let packed = pack_batch(&dense.view());
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], BitSdr::from_dense_row(&[1.0, 0.0, 0.6, 0.4]));
        assert_eq!(packed[0].active_bits(), 2);
        assert_eq!(packed[1].active_bits(), 2);
    }
}
