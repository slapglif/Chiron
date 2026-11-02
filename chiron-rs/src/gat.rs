//! GATv2 graph attention layer (Brody et al., 2022), inference path.
//!
//! Port of `chiron/layers/snn/graph_attention.py::GraphAttentionLayer`.
//! GATv2 dynamic attention: `e_ij = a_h^T LeakyReLU(q_i^h + k_j^h)`, scaled
//! by `1/sqrt(d_k)`, masked by the adjacency, softmaxed over neighbours j;
//! `out_i = sum_j alpha_ij v_j^h`, heads concatenated (or averaged), then an
//! output projection and a residual connection (pre-norm architecture).
//!
//! Optimizations relative to the Python reference:
//!
//! - **Edge-only sparse attention.** The Python materialises the full
//!   `(S, S)` logit matrix (tiled, but still O(S^2 * d_k) work) and then
//!   masks non-edges with `dtype.min`. Here, when a CSR adjacency is given,
//!   scores are computed *only* for edges present in the matrix —
//!   O(nnz * d_k) work and O(max_degree) scratch per thread.
//! - **Fused per-head projection GEMMs.** The per-head weights `(H, in, d_k)`
//!   are stored as one fused `(in, H*d_k)` matrix per projection, so Q/K/V
//!   for all heads are three single GEMMs. Each head's `(in, d_k)` block is
//!   still Xavier-initialised with its per-head fan (fan_in = in_features,
//!   fan_out = d_k) and leaky-relu gain, matching the Python init.
//! - **No allocations in the per-edge inner loop**: per-row neighbour-score
//!   scratch buffers are thread-local (`for_each_init`) and reused.
//!
//! Documented deviation: rows with ZERO neighbours produce a zero attention
//! aggregation here. The Python masks non-edges with `dtype.min`, which for
//! a fully-masked (isolated) row degenerates into *uniform* attention over
//! all nodes after softmax — an artifact of the masking trick, not intent.
//! The isolated node's output is therefore the pure residual pathway:
//! `residual(x) + out_proj(0)`.
//!
//! Adjacency values are used only as a connectivity mask (entry present =>
//! edge), mirroring the Python `adj == 0` masking; edge weights do not bias
//! the scores. Dropout layers are the identity (inference).

use ndarray::parallel::prelude::*;
use ndarray::{s, Array2, Array3, ArrayView2, ArrayView3, Axis};
use rand::rngs::StdRng;

use crate::nn::{leaky_relu, leaky_relu_gain, softmax_slice, xavier_uniform, LayerNorm, Linear};
use crate::sparse::CsrMatrix;

/// Build a fused multi-head projection `(in_features, num_heads * d_k)` where
/// each head's `(in_features, d_k)` column-block is Xavier-uniform with
/// fan_in = `in_features`, fan_out = `d_k` and the given gain.
fn fused_head_projection(
    in_features: usize,
    num_heads: usize,
    d_k: usize,
    gain: f32,
    rng: &mut StdRng,
) -> Array2<f32> {
    let mut w = Array2::<f32>::zeros((in_features, num_heads * d_k));
    for h in 0..num_heads {
        let vals = xavier_uniform(in_features, d_k, gain, in_features * d_k, rng);
        for i in 0..in_features {
            for d in 0..d_k {
                w[(i, h * d_k + d)] = vals[i * d_k + d];
            }
        }
    }
    w
}

/// GATv2 multi-head graph attention with Q/K/V projections, pre-norm,
/// output projection and residual connection.
pub struct GraphAttentionLayer {
    pub in_features: usize,
    pub out_features: usize,
    pub num_heads: usize,
    /// Per-head feature dimensionality.
    pub d_k: usize,
    /// LeakyReLU negative slope used in the attention scoring.
    pub alpha: f32,
    /// Concatenate head outputs (true) or average them (false).
    pub concat: bool,
    /// `1 / sqrt(d_k)`.
    scale: f32,
    /// Fused Q/K/V projections, each `(in_features, num_heads * d_k)`.
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    /// GATv2 attention vectors, `(num_heads, d_k)`.
    a: Array2<f32>,
    out_proj: Linear,
    layer_norm: LayerNorm,
    /// Linear skip projection when `in_features != out_features`.
    residual_proj: Option<Linear>,
}

impl GraphAttentionLayer {
    /// RNG draw order: `w_q`, `w_k`, `w_v`, `a`, `out_proj`,
    /// `residual_proj` (if any).
    pub fn new(
        in_features: usize,
        out_features: usize,
        num_heads: usize,
        alpha: f32,
        concat: bool,
        rng: &mut StdRng,
    ) -> Self {
        let d_k = if concat {
            assert!(
                out_features.is_multiple_of(num_heads),
                "out_features ({out_features}) must be divisible by num_heads \
                 ({num_heads}) when concat=true"
            );
            out_features / num_heads
        } else {
            out_features
        };
        let gain = leaky_relu_gain(alpha);

        let w_q = fused_head_projection(in_features, num_heads, d_k, gain, rng);
        let w_k = fused_head_projection(in_features, num_heads, d_k, gain, rng);
        let w_v = fused_head_projection(in_features, num_heads, d_k, gain, rng);
        let a = Array2::from_shape_vec(
            (num_heads, d_k),
            xavier_uniform(d_k, 1, gain, num_heads * d_k, rng),
        )
        .expect("attention vector shape");

        let proj_in = if concat { num_heads * d_k } else { d_k };
        let out_proj = Linear::new(proj_in, out_features, true, rng);
        let residual_proj =
            (in_features != out_features).then(|| Linear::new(in_features, out_features, true, rng));

        Self {
            in_features,
            out_features,
            num_heads,
            d_k,
            alpha,
            concat,
            scale: 1.0 / (d_k as f32).sqrt(),
            w_q,
            w_k,
            w_v,
            a,
            out_proj,
            layer_norm: LayerNorm::new(in_features),
            residual_proj,
        }
    }

    /// Attention aggregation: project the (already layer-normed) features to
    /// Q/K/V and return the per-head aggregated values `(S, num_heads * d_k)`
    /// (heads laid out as contiguous d_k blocks).
    ///
    /// Sparse path: scores only over CSR edges, O(nnz * d_k); rows with no
    /// neighbours stay zero. Dense path (`adj = None`): all-to-all attention
    /// with O(S) scratch per thread, streaming over j.
    fn attend(&self, xn: &ArrayView2<f32>, adj: Option<&CsrMatrix>) -> Array2<f32> {
        let s_len = xn.nrows();
        let heads = self.num_heads;
        let d_k = self.d_k;
        let hd = heads * d_k;
        let alpha = self.alpha;
        let scale = self.scale;

        // Three fused GEMMs cover all heads.
        let q = xn.dot(&self.w_q);
        let k = xn.dot(&self.w_k);
        let v = xn.dot(&self.w_v);
        let qs = q.as_slice().expect("Q contiguous");
        let ks = k.as_slice().expect("K contiguous");
        let vs = v.as_slice().expect("V contiguous");
        let av = self.a.as_slice().expect("a contiguous"); // (H, d_k) row-major

        let mut agg = Array2::<f32>::zeros((s_len, hd));

        // GATv2 logit for one (i, j, head) triple.
        #[inline(always)]
        fn edge_score(
            q_row: &[f32],
            k_row: &[f32],
            a_row: &[f32],
            alpha: f32,
            scale: f32,
        ) -> f32 {
            let mut e = 0.0_f32;
            for d in 0..a_row.len() {
                e += a_row[d] * leaky_relu(q_row[d] + k_row[d], alpha);
            }
            e * scale
        }

        match adj {
            // ---- (a) sparse path: per-edge scores only ----
            Some(csr) => {
                assert_eq!(csr.rows, s_len, "adjacency rows must match seq_len");
                assert_eq!(csr.cols, s_len, "adjacency cols must match seq_len");
                let max_deg = (0..s_len)
                    .map(|i| csr.indptr[i + 1] - csr.indptr[i])
                    .max()
                    .unwrap_or(0);

                agg.axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each_init(
                        || Vec::<f32>::with_capacity(max_deg),
                        |scores, (i, mut out_row)| {
                            let (cols, _vals) = csr.row(i);
                            if cols.is_empty() {
                                // Isolated node: zero aggregation (see module
                                // docs for the deviation from the Python).
                                return;
                            }
                            let out = out_row.as_slice_mut().unwrap();
                            for h in 0..heads {
                                let hoff = h * d_k;
                                let q_row = &qs[i * hd + hoff..i * hd + hoff + d_k];
                                let a_row = &av[hoff..hoff + d_k];
                                scores.clear();
                                for &j in cols {
                                    let k_row = &ks[j * hd + hoff..j * hd + hoff + d_k];
                                    scores.push(edge_score(q_row, k_row, a_row, alpha, scale));
                                }
                                softmax_slice(scores);
                                let o = &mut out[hoff..hoff + d_k];
                                for (&w, &j) in scores.iter().zip(cols) {
                                    let v_row = &vs[j * hd + hoff..j * hd + hoff + d_k];
                                    for d in 0..d_k {
                                        o[d] += w * v_row[d];
                                    }
                                }
                            }
                        },
                    );
            }
            // ---- (b) dense path: all-to-all attention ----
            None => {
                agg.axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each_init(
                        || Vec::<f32>::with_capacity(s_len),
                        |scores, (i, mut out_row)| {
                            let out = out_row.as_slice_mut().unwrap();
                            for h in 0..heads {
                                let hoff = h * d_k;
                                let q_row = &qs[i * hd + hoff..i * hd + hoff + d_k];
                                let a_row = &av[hoff..hoff + d_k];
                                // Pass 1: stream over j, scores into O(S) scratch.
                                scores.clear();
                                for j in 0..s_len {
                                    let k_row = &ks[j * hd + hoff..j * hd + hoff + d_k];
                                    scores.push(edge_score(q_row, k_row, a_row, alpha, scale));
                                }
                                softmax_slice(scores);
                                // Pass 2: stream over j again, accumulate V.
                                let o = &mut out[hoff..hoff + d_k];
                                for (j, &w) in scores.iter().enumerate() {
                                    let v_row = &vs[j * hd + hoff..j * hd + hoff + d_k];
                                    for d in 0..d_k {
                                        o[d] += w * v_row[d];
                                    }
                                }
                            }
                        },
                    );
            }
        }
        agg
    }

    /// Single-graph forward: `x (S, in_features) -> (S, out_features)`.
    ///
    /// Pre-norm residual composition:
    /// `out = residual(x) + out_proj(attend(layer_norm(x), adj))`.
    pub fn forward(&self, x: &ArrayView2<f32>, adj: Option<&CsrMatrix>) -> Array2<f32> {
        debug_assert_eq!(x.ncols(), self.in_features);
        let xn = self.layer_norm.forward(x);
        let agg = self.attend(&xn.view(), adj);

        // Multi-head combination: concat keeps the (S, H*d_k) layout as-is;
        // averaging collapses the H d_k-blocks.
        let combined = if self.concat {
            agg
        } else {
            let s_len = agg.nrows();
            let mut mean = Array2::<f32>::zeros((s_len, self.d_k));
            for h in 0..self.num_heads {
                mean += &agg.slice(s![.., h * self.d_k..(h + 1) * self.d_k]);
            }
            mean /= self.num_heads as f32;
            mean
        };

        let mut out = self.out_proj.forward(&combined.view());
        match &self.residual_proj {
            Some(p) => out += &p.forward(x),
            None => out += x,
        }
        out
    }

    /// Batched forward: `x (B, S, in) -> (B, S, out)`, sharing one adjacency
    /// across the batch. Serial over B by design — the inner per-row work is
    /// already rayon-parallel, and nesting a par_iter over the batch would
    /// oversubscribe the thread pool.
    pub fn forward_batch(&self, x: &ArrayView3<f32>, adj: Option<&CsrMatrix>) -> Array3<f32> {
        let (b, s_len, _) = x.dim();
        let mut out = Array3::<f32>::zeros((b, s_len, self.out_features));
        for bi in 0..b {
            let xi = x.index_axis(Axis(0), bi);
            let yi = self.forward(&xi, adj);
            out.index_axis_mut(Axis(0), bi).assign(&yi);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::seeded_rng;

    fn sample_x(rows: usize, cols: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |(i, j)| {
            ((i * 13 + j * 5) as f32 * 0.29).sin()
        })
    }

    fn max_abs_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn output_shape_concat_true() {
        let mut rng = seeded_rng(10);
        // in != out exercises the residual projection.
        let layer = GraphAttentionLayer::new(12, 8, 2, 0.2, true, &mut rng);
        assert_eq!(layer.d_k, 4);
        let x = sample_x(5, 12);
        let y = layer.forward(&x.view(), None);
        assert_eq!(y.dim(), (5, 8));
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn output_shape_concat_false() {
        let mut rng = seeded_rng(11);
        let layer = GraphAttentionLayer::new(8, 5, 3, 0.2, false, &mut rng);
        assert_eq!(layer.d_k, 5);
        let x = sample_x(6, 8);
        let y = layer.forward(&x.view(), None);
        assert_eq!(y.dim(), (6, 5));
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_batch_shape() {
        let mut rng = seeded_rng(12);
        let layer = GraphAttentionLayer::new(8, 8, 2, 0.2, true, &mut rng);
        let x = Array3::from_shape_fn((3, 4, 8), |(b, s, f)| {
            ((b * 17 + s * 7 + f) as f32 * 0.11).cos()
        });
        let dense = Array2::<f32>::ones((4, 4));
        let csr = CsrMatrix::from_dense(&dense.view());
        let y = layer.forward_batch(&x.view(), Some(&csr));
        assert_eq!(y.dim(), (3, 4, 8));
        // Batch element bi must equal the single-graph forward on its slice.
        for bi in 0..3 {
            let yi = layer.forward(&x.index_axis(Axis(0), bi), Some(&csr));
            assert_eq!(y.index_axis(Axis(0), bi), yi);
        }
    }

    #[test]
    fn single_neighbor_aggregation_equals_value_row() {
        let mut rng = seeded_rng(13);
        let layer = GraphAttentionLayer::new(8, 8, 2, 0.2, true, &mut rng);
        // Ring graph: row i has exactly one neighbour (i+1) % 4, so softmax
        // over the neighbourhood is the constant 1 and the aggregation must
        // be exactly that neighbour's V row (attention weights sum to 1).
        let csr = CsrMatrix {
            rows: 4,
            cols: 4,
            indptr: vec![0, 1, 2, 3, 4],
            indices: vec![1, 2, 3, 0],
            data: vec![1.0; 4],
        };
        let xn = sample_x(4, 8);
        let agg = layer.attend(&xn.view(), Some(&csr));
        let v = xn.dot(&layer.w_v);
        for i in 0..4 {
            let j = (i + 1) % 4;
            for d in 0..8 {
                assert!(
                    (agg[(i, d)] - v[(j, d)]).abs() < 1e-5,
                    "row {i} dim {d}: agg={} v[{j}]={}",
                    agg[(i, d)],
                    v[(j, d)]
                );
            }
        }
    }

    #[test]
    fn isolated_node_takes_pure_residual_pathway() {
        let mut rng = seeded_rng(14);
        // in == out: residual is the raw input, no residual projection.
        let layer = GraphAttentionLayer::new(8, 8, 2, 0.2, true, &mut rng);
        // Row 0 has no neighbours; the rest attend to node 0.
        let csr = CsrMatrix {
            rows: 4,
            cols: 4,
            indptr: vec![0, 0, 1, 2, 3],
            indices: vec![0, 0, 0],
            data: vec![1.0; 3],
        };
        let x = sample_x(4, 8);
        let y = layer.forward(&x.view(), Some(&csr));
        assert!(y.iter().all(|v| v.is_finite()));

        // Isolated node: aggregation is zero, so the output is
        // x[0] + out_proj(0) (i.e. the out_proj bias).
        let zeros = Array2::<f32>::zeros((1, layer.num_heads * layer.d_k));
        let bias_row = layer.out_proj.forward(&zeros.view());
        for d in 0..8 {
            let expected = x[(0, d)] + bias_row[(0, d)];
            assert!(
                (y[(0, d)] - expected).abs() < 1e-5,
                "dim {d}: got {} expected {expected}",
                y[(0, d)]
            );
        }
    }

    #[test]
    fn dense_path_matches_sparse_full_adjacency() {
        let mut rng = seeded_rng(15);
        let layer = GraphAttentionLayer::new(8, 8, 2, 0.2, true, &mut rng);
        let x = sample_x(6, 8);
        let dense = Array2::<f32>::ones((6, 6));
        let csr = CsrMatrix::from_dense(&dense.view());
        assert_eq!(csr.nnz(), 36);
        let y_sparse = layer.forward(&x.view(), Some(&csr));
        let y_dense = layer.forward(&x.view(), None);
        let diff = max_abs_diff(&y_sparse, &y_dense);
        assert!(diff < 1e-4, "dense vs sparse mismatch: {diff}");
    }

    #[test]
    fn deterministic_given_seed() {
        let x = sample_x(5, 12);
        let dense = Array2::<f32>::ones((5, 5));
        let csr = CsrMatrix::from_dense(&dense.view());
        let mut rng_a = seeded_rng(99);
        let mut rng_b = seeded_rng(99);
        let a = GraphAttentionLayer::new(12, 8, 2, 0.2, true, &mut rng_a);
        let b = GraphAttentionLayer::new(12, 8, 2, 0.2, true, &mut rng_b);
        let ya = a.forward(&x.view(), Some(&csr));
        let yb = b.forward(&x.view(), Some(&csr));
        assert_eq!(ya, yb);
        // Same layer, repeated call: reproducible despite rayon parallelism
        // (each output row is accumulated sequentially).
        assert_eq!(ya, a.forward(&x.view(), Some(&csr)));
    }
}
