//! Shared neural-network primitives: Linear, LayerNorm, activations, RNG init.
//!
//! All tensors are `f32`, row-major `ndarray` arrays. Matrix products go
//! through `ndarray::dot` (matrixmultiply: blocked, SIMD-friendly GEMM).
//! Batch-parallel ops use rayon over rows.

use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2, Axis};
use ndarray::parallel::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Deterministic RNG used everywhere so runs are reproducible (seed in config).
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Fully-connected layer. Weight is stored transposed relative to PyTorch,
/// as `(in_features, out_features)`, so the forward pass is a single GEMM
/// `x (n, in) @ w (in, out)` with no per-call transpose.
pub struct Linear {
    pub weight: Array2<f32>, // (in_features, out_features)
    pub bias: Option<Array1<f32>>,
}

impl Linear {
    /// Xavier-uniform initialisation (PyTorch `nn.Linear` default family).
    pub fn new(in_features: usize, out_features: usize, bias: bool, rng: &mut StdRng) -> Self {
        let bound = (6.0_f32 / (in_features + out_features) as f32).sqrt();
        let dist = Uniform::new_inclusive(-bound, bound);
        let weight =
            Array2::from_shape_fn((in_features, out_features), |_| dist.sample(rng));
        let bias = bias.then(|| Array1::zeros(out_features));
        Self { weight, bias }
    }

    /// Truncated-normal style init used by the SDR input projector.
    pub fn new_normal(in_features: usize, out_features: usize, std: f32, rng: &mut StdRng) -> Self {
        let dist = Normal::new(0.0_f32, std).unwrap();
        let weight = Array2::from_shape_fn((in_features, out_features), |_| {
            // clamp to +/- 2 std: cheap truncated normal
            dist.sample(rng).clamp(-2.0 * std, 2.0 * std)
        });
        Self { weight, bias: None }
    }

    /// `x: (n, in) -> (n, out)`
    pub fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let mut out = x.dot(&self.weight);
        if let Some(b) = &self.bias {
            out.axis_iter_mut(Axis(0))
                .into_par_iter()
                .for_each(|mut row| row += b);
        }
        out
    }
}

/// Layer normalisation over the last axis of a 2-D batch.
pub struct LayerNorm {
    pub gamma: Array1<f32>,
    pub beta: Array1<f32>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: Array1::ones(dim),
            beta: Array1::zeros(dim),
            eps: 1e-5,
        }
    }

    /// Normalise each row in place: one pass for mean, one fused pass for
    /// variance + affine transform. Rayon-parallel over rows.
    pub fn forward_inplace(&self, mut x: ArrayViewMut2<f32>) {
        let gamma = self.gamma.as_slice().unwrap();
        let beta = self.beta.as_slice().unwrap();
        let eps = self.eps;
        x.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut row| {
            let row = row.as_slice_mut().unwrap();
            let n = row.len() as f32;
            let mean = row.iter().sum::<f32>() / n;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let inv_std = 1.0 / (var + eps).sqrt();
            for (i, v) in row.iter_mut().enumerate() {
                *v = (*v - mean) * inv_std * gamma[i] + beta[i];
            }
        });
    }

    pub fn forward(&self, x: &ArrayView2<f32>) -> Array2<f32> {
        let mut out = x.to_owned();
        self.forward_inplace(out.view_mut());
        out
    }
}

/// Row-wise softmax in place (numerically stable, max-subtracted).
pub fn softmax_rows_inplace(mut x: ArrayViewMut2<f32>) {
    x.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut row| {
        let row = row.as_slice_mut().unwrap();
        softmax_slice(row);
    });
}

/// Softmax over a contiguous slice (used for per-row / per-neighbourhood
/// normalisation in attention layers).
#[inline]
pub fn softmax_slice(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for v in row.iter_mut() {
        *v *= inv;
    }
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn softplus(x: f32) -> f32 {
    // Numerically stable: log(1 + e^x) = max(x, 0) + log(1 + e^{-|x|})
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

#[inline(always)]
pub fn leaky_relu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        alpha * x
    }
}

/// Tanh-approximation GELU (matches PyTorch `nn.GELU()` closely).
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

pub fn gelu_inplace(x: &mut Array2<f32>) {
    x.par_mapv_inplace(gelu);
}

/// In-row threshold of the k-th largest value, O(n) via select_nth_unstable.
/// Returns the value such that exactly the top-k elements are >= it
/// (ties broken pessimistically, like `torch.topk` threshold semantics).
pub fn topk_threshold(row: &[f32], k: usize) -> f32 {
    debug_assert!(k >= 1 && k <= row.len());
    let mut scratch: Vec<f32> = row.to_vec();
    let idx = k - 1;
    let (_, kth, _) = scratch
        .select_nth_unstable_by(idx, |a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    *kth
}

/// Xavier-uniform init for an arbitrary-shape weight with explicit fan-in/out
/// and gain (used by GAT per-head parameter tensors).
pub fn xavier_uniform(shape_fan_in: usize, shape_fan_out: usize, gain: f32, len: usize, rng: &mut StdRng) -> Vec<f32> {
    let bound = gain * (6.0_f32 / (shape_fan_in + shape_fan_out) as f32).sqrt();
    let dist = Uniform::new_inclusive(-bound, bound);
    (0..len).map(|_| dist.sample(rng)).collect()
}

/// Gain for leaky_relu as computed by `torch.nn.init.calculate_gain`.
pub fn leaky_relu_gain(negative_slope: f32) -> f32 {
    (2.0 / (1.0 + negative_slope * negative_slope)).sqrt()
}

/// Orthogonal-ish init for recurrent weights: random normal followed by
/// row Gram-Schmidt. For the small square matrices used here (<= 256x256)
/// this is exact and fast.
pub fn orthogonal(dim: usize, gain: f32, rng: &mut StdRng) -> Array2<f32> {
    let normal = Normal::new(0.0_f32, 1.0).unwrap();
    let mut m: Vec<Vec<f32>> = (0..dim)
        .map(|_| (0..dim).map(|_| normal.sample(rng)).collect())
        .collect();
    for i in 0..dim {
        let (prev, rest) = m.split_at_mut(i);
        let row_i = &mut rest[0];
        for row_j in prev.iter() {
            let dot: f32 = row_i.iter().zip(row_j).map(|(a, b)| a * b).sum();
            for (a, b) in row_i.iter_mut().zip(row_j) {
                *a -= dot * b;
            }
        }
        let norm: f32 = row_i.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in row_i.iter_mut() {
            *v = *v / norm * gain;
        }
    }
    Array2::from_shape_fn((dim, dim), |(i, j)| m[i][j])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn linear_shapes_and_bias() {
        let mut rng = seeded_rng(0);
        let lin = Linear::new(4, 3, true, &mut rng);
        let x = Array2::<f32>::ones((2, 4));
        let y = lin.forward(&x.view());
        assert_eq!(y.shape(), &[2, 3]);
    }

    #[test]
    fn layernorm_zero_mean_unit_var() {
        let ln = LayerNorm::new(4);
        let x = array![[1.0_f32, 2.0, 3.0, 4.0]];
        let y = ln.forward(&x.view());
        let mean: f32 = y.row(0).sum() / 4.0;
        let var: f32 = y.row(0).iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let mut x = array![[1.0_f32, 2.0, 3.0], [0.0, 0.0, 0.0]];
        softmax_rows_inplace(x.view_mut());
        for row in x.rows() {
            assert!((row.sum() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn topk_threshold_selects_kth() {
        let row = [0.1_f32, 0.9, 0.5, 0.7];
        assert_eq!(topk_threshold(&row, 1), 0.9);
        assert_eq!(topk_threshold(&row, 2), 0.7);
        assert_eq!(topk_threshold(&row, 4), 0.1);
    }

    #[test]
    fn orthogonal_rows_are_orthonormal() {
        let mut rng = seeded_rng(7);
        let m = orthogonal(8, 1.0, &mut rng);
        for i in 0..8 {
            for j in 0..8 {
                let dot: f32 = m.row(i).dot(&m.row(j));
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-4, "rows {i},{j} dot={dot}");
            }
        }
    }
}
