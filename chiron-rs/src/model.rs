//! Full pipeline model: Rust port of `chiron/layers/snn/model.py::SNNModel`.
//!
//! Wires the layers in the exact order of the Python forward pass:
//! input projection -> SNN -> temporal attention -> GATv2 (+ skip) ->
//! attention mask -> HTM (+ skip) -> node gather -> MLP head.
//!
//! All inter-layer activations are kept as flat `(batch * seq, feat)`
//! matrices so every projection is a single GEMM; the only 3-D reshape is
//! the zero-copy view handed to the GAT (which needs per-graph structure)
//! and the SNN timestep cube.

use ndarray::{Array2, Array3, ArrayView2, Axis};
use ndarray::parallel::prelude::*;
use rand::rngs::StdRng;

use crate::gat::GraphAttentionLayer;
use crate::htm::{HtmModel, HtmParams};
use crate::nn::{seeded_rng, LayerNorm, Linear};
use crate::snn::{MlpHead, SnnLayer, TemporalAttention};
use crate::sparse::CsrMatrix;

/// Hyper-parameters, defaults mirroring `chiron/config.json`.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub sdr_dimensions: usize,
    pub snn_input_size: usize,
    pub snn_hidden_size: usize,
    pub snn_output_size: usize,
    pub snn_timesteps: usize,
    pub gat_out_features: usize,
    pub gat_num_heads: usize,
    pub gat_alpha: f32,
    pub gat_concat: bool,
    pub htm: HtmParams,
    pub seed: u64,
    pub sparsity: SparsityConfig,
}

/// Controls the sparse / approximate execution paths.
///
/// The pipeline's intermediate signal is dominated by the SNN spike cube
/// `(N, T, 128)`. Spikes are binary and LayerNorm of a binary vector takes
/// only two values per row, so the cube can be carried as bit-packed masks
/// (~30x smaller) and every downstream projection becomes a sparse
/// accumulation of weight rows plus a rank-1 correction — exactly.
/// The only *approximation* is `recurrent_keep < 1.0`, which magnitude-prunes
/// the recurrent membrane weights (the one inherently dense GEMM left).
#[derive(Clone, Copy, Debug)]
pub struct SparsityConfig {
    /// Run the SNN + temporal attention via bit-packed spike events.
    pub event_driven_snn: bool,
    /// Fraction of recurrent weights kept (by magnitude). 1.0 = exact.
    pub recurrent_keep: f32,
    /// Rows per chunk for the fused input-proj -> SNN -> temporal-attention
    /// stage; bounds the live spike-cube working set. 0 disables chunking.
    pub chunk_rows: usize,
}

impl SparsityConfig {
    /// Bit-exact reference execution (dense paths everywhere).
    pub fn exact() -> Self {
        Self {
            event_driven_snn: false,
            recurrent_keep: 1.0,
            chunk_rows: 0,
        }
    }

    /// Everything that is *exactly* sparsifiable, sparsified: bit-packed
    /// spike events, popcount-derived LayerNorm, sparse-accumulated
    /// projections, chunked execution. The recurrent membrane GEMM stays
    /// dense — measured on CPU, a CSR matvec at 50% kept weights *loses*
    /// to the SIMD dense GEMM (gather-bound), and the keep ratios where
    /// CSR wins (<= ~25%) cost too much fidelity for a default.
    /// Output matches [`Self::exact`] to float-summation tolerance.
    pub fn aggressive() -> Self {
        Self {
            event_driven_snn: true,
            recurrent_keep: 1.0,
            chunk_rows: 2048,
        }
    }

    /// [`Self::aggressive`] plus magnitude-pruned recurrent weights at 25%
    /// kept entries — the density where the CSR matvec actually beats the
    /// dense GEMM on CPU. This is a lossy approximation: ~72% of the
    /// recurrent matrix energy is retained and the output cosine vs exact
    /// is ~0.85 (the discrete spike threshold amplifies drive error).
    pub fn extreme() -> Self {
        Self {
            event_driven_snn: true,
            recurrent_keep: 0.25,
            chunk_rows: 2048,
        }
    }
}

impl Default for SparsityConfig {
    fn default() -> Self {
        Self::exact()
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            sdr_dimensions: 256,
            snn_input_size: 256,
            snn_hidden_size: 64,
            snn_output_size: 128,
            snn_timesteps: 10,
            gat_out_features: 64,
            gat_num_heads: 2,
            gat_alpha: 0.2,
            gat_concat: true,
            htm: HtmParams::default(),
            seed: 42,
            sparsity: SparsityConfig::default(),
        }
    }
}

/// Per-layer timing of one forward pass, in seconds. Filled by
/// [`SnnModel::forward_timed`]; used by the benchmark binary.
#[derive(Clone, Copy, Debug, Default)]
pub struct LayerTimings {
    pub input_projection: f64,
    pub snn: f64,
    pub temporal_attention: f64,
    pub gat: f64,
    pub htm: f64,
    pub head: f64,
    pub total: f64,
}

pub struct SnnModel {
    cfg: ModelConfig,
    input_projection: Linear, // (1 -> snn_input_size)
    input_ln: LayerNorm,
    snn_layer: SnnLayer,
    temporal_attention: TemporalAttention,
    post_snn_ln: LayerNorm,
    gat_layer: GraphAttentionLayer,
    post_gat_ln: LayerNorm,
    skip_snn_to_gat: Option<Linear>,
    htm_layer: HtmModel,
    post_htm_ln: LayerNorm,
    skip_gat_to_htm: Option<Linear>,
    mlp_head: MlpHead,
}

impl SnnModel {
    pub fn new(cfg: ModelConfig) -> Self {
        let mut rng: StdRng = seeded_rng(cfg.seed);
        let gat_out_dim = cfg.gat_out_features;
        let output_size = cfg.sdr_dimensions;

        let input_projection = Linear::new(1, cfg.snn_input_size, true, &mut rng);
        let input_ln = LayerNorm::new(cfg.snn_input_size);

        let snn_layer = SnnLayer::new(
            cfg.snn_input_size,
            cfg.snn_hidden_size,
            cfg.snn_output_size,
            cfg.snn_timesteps,
            &mut rng,
        );
        let temporal_attention = TemporalAttention::new(cfg.snn_output_size, &mut rng);
        let post_snn_ln = LayerNorm::new(cfg.snn_output_size);

        let gat_layer = GraphAttentionLayer::new(
            cfg.snn_output_size,
            gat_out_dim,
            cfg.gat_num_heads,
            cfg.gat_alpha,
            cfg.gat_concat,
            &mut rng,
        );
        let post_gat_ln = LayerNorm::new(gat_out_dim);
        let skip_snn_to_gat = (cfg.snn_output_size != gat_out_dim)
            .then(|| Linear::new(cfg.snn_output_size, gat_out_dim, true, &mut rng));

        let htm_layer = HtmModel::new(output_size, &cfg.htm, &mut rng);
        let post_htm_ln = LayerNorm::new(output_size);
        let skip_gat_to_htm = (gat_out_dim != output_size)
            .then(|| Linear::new(gat_out_dim, output_size, true, &mut rng));

        let mlp_hidden = (output_size / 2).max(64);
        let mlp_head = MlpHead::new(output_size, mlp_hidden, output_size, &mut rng);

        let mut snn_layer = snn_layer;
        if cfg.sparsity.recurrent_keep < 1.0 {
            snn_layer.prune_recurrent(cfg.sparsity.recurrent_keep);
        }

        Self {
            cfg,
            input_projection,
            input_ln,
            snn_layer,
            temporal_attention,
            post_snn_ln,
            gat_layer,
            post_gat_ln,
            skip_snn_to_gat,
            htm_layer,
            post_htm_ln,
            skip_gat_to_htm,
            mlp_head,
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// Forward pass.
    ///
    /// Args mirror the Python signature:
    /// - `input_ids`: `(batch, seq)` token ids (as f32, like the Python
    ///   `.float()` cast).
    /// - `attention_mask`: `(batch, seq)` of 0/1.
    /// - `adjacency`: optional CSR adjacency; the top-left `(seq, seq)`
    ///   principal submatrix is used, matching `adj[:seq_len, :seq_len]`.
    /// - `node_indices`: per-sample sequence position to read out, len batch.
    ///
    /// Returns `(batch, sdr_dimensions)`.
    pub fn forward(
        &mut self,
        input_ids: &ArrayView2<f32>,
        attention_mask: &ArrayView2<f32>,
        adjacency: Option<&CsrMatrix>,
        node_indices: &[usize],
        training: bool,
    ) -> Array2<f32> {
        self.forward_timed(input_ids, attention_mask, adjacency, node_indices, training)
            .0
    }

    /// Forward pass that also returns wall-clock per-layer timings.
    pub fn forward_timed(
        &mut self,
        input_ids: &ArrayView2<f32>,
        attention_mask: &ArrayView2<f32>,
        adjacency: Option<&CsrMatrix>,
        node_indices: &[usize],
        training: bool,
    ) -> (Array2<f32>, LayerTimings) {
        use std::time::Instant;
        let (batch, seq) = input_ids.dim();
        assert_eq!(attention_mask.dim(), (batch, seq), "attention mask shape");
        assert_eq!(node_indices.len(), batch, "node_indices length");
        let flat = batch * seq;
        let mut t = LayerTimings::default();
        let t_all = Instant::now();

        let ids_col = input_ids
            .to_owned()
            .into_shape_with_order((flat, 1))
            .expect("input reshape");

        // Stages 1-3: input embedding -> SNN -> temporal attention.
        //
        // Event-driven path: the three stages are fused and run chunk by
        // chunk so the live spike representation never exceeds
        // chunk_rows * T * dim/8 bytes (bit-packed), instead of the dense
        // (flat, T, dim) f32 cube. Each chunk's spikes flow straight into
        // temporal attention while still hot in cache.
        let mut snn_output = if self.cfg.sparsity.event_driven_snn {
            let chunk = match self.cfg.sparsity.chunk_rows {
                0 => flat,
                c => c,
            };
            let mut out = Array2::<f32>::zeros((flat, self.cfg.snn_output_size));
            let mut start = 0;
            while start < flat {
                let end = (start + chunk).min(flat);
                let ids_chunk = ids_col.slice(ndarray::s![start..end, ..]);

                let t0 = Instant::now();
                let mut x = self.input_projection.forward(&ids_chunk);
                self.input_ln.forward_inplace(x.view_mut());
                t.input_projection += t0.elapsed().as_secs_f64();

                let t0 = Instant::now();
                let spikes = self.snn_layer.forward_events(&x.view());
                t.snn += t0.elapsed().as_secs_f64();

                let t0 = Instant::now();
                let ta = self
                    .temporal_attention
                    .forward_events(&spikes, self.snn_layer.ln2());
                out.slice_mut(ndarray::s![start..end, ..]).assign(&ta);
                t.temporal_attention += t0.elapsed().as_secs_f64();

                start = end;
            }
            out
        } else {
            // 1. Learned input embedding: each scalar id -> vector.
            let t0 = Instant::now();
            let mut x = self.input_projection.forward(&ids_col.view()); // (flat, in)
            self.input_ln.forward_inplace(x.view_mut());
            t.input_projection = t0.elapsed().as_secs_f64();

            // 2. SNN layer over timesteps.
            let t0 = Instant::now();
            let snn_raw: Array3<f32> = self.snn_layer.forward(&x.view()); // (flat, T, out)
            t.snn = t0.elapsed().as_secs_f64();

            // 3. Temporal attention collapses the timestep axis.
            let t0 = Instant::now();
            let out = self.temporal_attention.forward(&snn_raw.view()); // (flat, out)
            t.temporal_attention = t0.elapsed().as_secs_f64();
            out
        };
        let t0 = Instant::now();
        self.post_snn_ln.forward_inplace(snn_output.view_mut());
        t.temporal_attention += t0.elapsed().as_secs_f64();

        // 4. GATv2 over the adjacency graph (per batch element).
        let t0 = Instant::now();
        let snn_3d = snn_output
            .view()
            .into_shape_with_order((batch, seq, self.cfg.snn_output_size))
            .expect("snn reshape");
        let sliced;
        let adj_for_gat = match adjacency {
            Some(a) if a.rows > seq => {
                sliced = a.principal_submatrix(seq);
                Some(&sliced)
            }
            other => other,
        };
        let gat_3d = self.gat_layer.forward_batch(&snn_3d, adj_for_gat); // (B, S, gat_out)
        let gat_out_dim = gat_3d.dim().2;
        let mut gat_output = gat_3d
            .into_shape_with_order((flat, gat_out_dim))
            .expect("gat reshape");

        // Residual skip from SNN into the GAT output.
        let snn_skip = match &self.skip_snn_to_gat {
            Some(proj) => proj.forward(&snn_output.view()),
            None => snn_output,
        };
        gat_output += &snn_skip;
        self.post_gat_ln.forward_inplace(gat_output.view_mut());

        // 5. Attention mask: zero out padded positions.
        let mask_flat = attention_mask
            .to_owned()
            .into_shape_with_order(flat)
            .expect("mask reshape");
        gat_output
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                let m = mask_flat[i];
                if m != 1.0 {
                    row.mapv_inplace(|v| v * m);
                }
            });
        t.gat = t0.elapsed().as_secs_f64();

        // 6. HTM spatial pooler (+ skip from GAT).
        let t0 = Instant::now();
        let mut htm_output = self.htm_layer.forward(&gat_output.view(), training); // (flat, sdr)
        let gat_skip = match &self.skip_gat_to_htm {
            Some(proj) => proj.forward(&gat_output.view()),
            None => gat_output,
        };
        htm_output += &gat_skip;
        self.post_htm_ln.forward_inplace(htm_output.view_mut());
        t.htm = t0.elapsed().as_secs_f64();

        // 7. Gather the requested node per sample and run the MLP head.
        let t0 = Instant::now();
        let mut selected = Array2::<f32>::zeros((batch, self.cfg.sdr_dimensions));
        for (b, mut row) in selected.axis_iter_mut(Axis(0)).enumerate() {
            let idx = node_indices[b];
            assert!(idx < seq, "node index {idx} out of range for seq {seq}");
            row.assign(&htm_output.row(b * seq + idx));
        }
        let output = self.mlp_head.forward(&selected.view());
        t.head = t0.elapsed().as_secs_f64();

        t.total = t_all.elapsed().as_secs_f64();
        (output, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn small_cfg() -> ModelConfig {
        ModelConfig::default()
    }

    #[test]
    fn end_to_end_shapes_and_determinism() {
        let (b, s) = (4, 16);
        let ids = Array2::from_shape_fn((b, s), |(i, j)| ((i * 31 + j * 7) % 997) as f32);
        let mask = Array2::<f32>::ones((b, s));
        let nodes = vec![0usize; b];

        let mut m1 = SnnModel::new(small_cfg());
        let mut m2 = SnnModel::new(small_cfg());
        let y1 = m1.forward(&ids.view(), &mask.view(), None, &nodes, false);
        let y2 = m2.forward(&ids.view(), &mask.view(), None, &nodes, false);
        assert_eq!(y1.dim(), (b, 256));
        assert!(y1.iter().all(|v| v.is_finite()));
        assert_eq!(y1, y2, "same seed must give identical outputs");
    }

    #[test]
    fn adjacency_is_sliced_to_seq_len() {
        let (b, s) = (2, 8);
        let ids = Array2::from_shape_fn((b, s), |(i, j)| (i + j) as f32);
        let mask = Array2::<f32>::ones((b, s));
        let nodes = vec![3usize; b];

        // Adjacency bigger than seq: ring graph over 32 nodes.
        let mut dense = Array2::<f32>::zeros((32, 32));
        for i in 0..32 {
            dense[(i, (i + 1) % 32)] = 1.0;
            dense[(i, (i + 31) % 32)] = 1.0;
        }
        let csr = CsrMatrix::from_dense(&dense.view());

        let mut m = SnnModel::new(small_cfg());
        let y = m.forward(&ids.view(), &mask.view(), Some(&csr), &nodes, false);
        assert_eq!(y.dim(), (b, 256));
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn event_driven_path_matches_dense_path() {
        let (b, s) = (4, 16);
        let ids = Array2::from_shape_fn((b, s), |(i, j)| ((i * 17 + j * 5) % 503) as f32);
        let mask = Array2::<f32>::ones((b, s));
        let nodes = vec![1usize; b];

        let mut dense = SnnModel::new(small_cfg());
        let mut sparse_cfg = small_cfg();
        sparse_cfg.sparsity = SparsityConfig {
            event_driven_snn: true,
            recurrent_keep: 1.0, // no approximation: must match the dense path
            chunk_rows: 16,      // force several chunks
        };
        let mut sparse = SnnModel::new(sparse_cfg);

        let yd = dense.forward(&ids.view(), &mask.view(), None, &nodes, false);
        let ys = sparse.forward(&ids.view(), &mask.view(), None, &nodes, false);
        let max_abs = yd
            .iter()
            .zip(ys.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs < 1e-3,
            "event-driven path deviates from dense: max abs diff {max_abs}"
        );
    }

    #[test]
    fn aggressive_sparsity_stays_faithful() {
        let (b, s) = (4, 16);
        let ids = Array2::from_shape_fn((b, s), |(i, j)| ((i * 29 + j * 3) % 701) as f32);
        let mask = Array2::<f32>::ones((b, s));
        let nodes = vec![0usize; b];

        let mut dense = SnnModel::new(small_cfg());
        let mut agg_cfg = small_cfg();
        agg_cfg.sparsity = SparsityConfig::aggressive();
        let mut sparse = SnnModel::new(agg_cfg);
        let mut ext_cfg = small_cfg();
        ext_cfg.sparsity = SparsityConfig::extreme();
        let mut extreme = SnnModel::new(ext_cfg);

        let cosine = |a: &Array2<f32>, b: &Array2<f32>| {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
            dot / (na * nb).max(1e-12)
        };

        let yd = dense.forward(&ids.view(), &mask.view(), None, &nodes, false);

        // aggressive() contains no approximation: must match near-exactly.
        let ya = sparse.forward(&ids.view(), &mask.view(), None, &nodes, false);
        let cos_a = cosine(&yd, &ya);
        assert!(cos_a > 0.999, "aggressive (exact-sparse) diverged: cosine {cos_a}");

        // extreme() prunes recurrent weights to 25%: lossy but bounded.
        let ye = extreme.forward(&ids.view(), &mask.view(), None, &nodes, false);
        let cos_e = cosine(&yd, &ye);
        assert!(cos_e > 0.8, "extreme sparsity output diverged: cosine {cos_e}");
    }

    #[test]
    fn training_mode_runs_htm_plasticity() {
        let (b, s) = (2, 8);
        let ids = Array2::from_shape_fn((b, s), |(i, j)| (i * 13 + j) as f32);
        let mask = Array2::<f32>::ones((b, s));
        let nodes = vec![0usize; b];
        let mut m = SnnModel::new(small_cfg());
        let y = m.forward(&ids.view(), &mask.view(), None, &nodes, true);
        assert!(y.iter().all(|v| v.is_finite()));
    }
}
