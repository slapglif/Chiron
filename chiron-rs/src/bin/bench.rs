//! Per-layer throughput benchmark for the chiron-core pipeline.
//!
//! Run with: `cargo run --release --bin bench`
//!
//! Measures, at the dimensions from `chiron/config.json`:
//!   1. SDR generation (FlyHash + kWTA) throughput
//!   2. Cosine kNN adjacency construction
//!   3. Bit-packed SDR similarity ops
//!   4. Every pipeline stage of `SnnModel::forward` (signal-flow throughput)

use std::time::Instant;

use ndarray::Array2;
use rand::Rng;

use chiron_core::model::{LayerTimings, ModelConfig, SnnModel, SparsityConfig};
use chiron_core::nn::seeded_rng;
use chiron_core::sdr::{pack_batch, SdrGenerator};
use chiron_core::sparse::CsrMatrix;

fn main() {
    println!("chiron-core throughput benchmark");
    println!(
        "threads: {}, build: release\n",
        rayon::current_num_threads()
    );

    bench_sdr();
    bench_adjacency();
    bench_bitsdr();
    bench_pipeline();
}

fn bench_sdr() {
    let mut rng = seeded_rng(1);
    let n = 8192;
    let dim = 300; // Word2Vec embedding width
    let emb = Array2::from_shape_fn((n, dim), |_| rng.gen_range(-1.0_f32..1.0));

    let mut gen = SdrGenerator::new(64, 256, 0.002, 42);
    // warmup (also builds the lazy input projector)
    let _ = gen.forward(&emb.view(), false);

    let t0 = Instant::now();
    let iters = 10;
    for _ in 0..iters {
        let sdr = gen.forward(&emb.view(), false);
        std::hint::black_box(&sdr);
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    println!("[SDR generation]  300d -> 64d -> 256d binary, sparsity 0.002");
    println!(
        "  {} embeddings in {:.2} ms  ->  {:.0} embeddings/sec\n",
        n,
        dt * 1e3,
        n as f64 / dt
    );
}

fn bench_adjacency() {
    let mut rng = seeded_rng(2);
    let n = 1024;
    // Clustered embeddings (32 centroids + noise) so the 0.5 cosine
    // threshold keeps realistic neighbourhoods; i.i.d. uniform vectors in
    // 256-d are near-orthogonal and would yield an empty adjacency.
    let centroids = Array2::from_shape_fn((32, 256), |_| rng.gen_range(-1.0_f32..1.0));
    let emb = Array2::from_shape_fn((n, 256), |(i, j)| {
        centroids[(i % 32, j)] + rng.gen_range(-0.4_f32..0.4)
    });

    let t0 = Instant::now();
    let adj = CsrMatrix::cosine_adjacency(&emb.view(), 0.5, 50);
    let dt = t0.elapsed().as_secs_f64();
    println!("[Adjacency build] cosine kNN, threshold 0.5, top_k 50");
    println!(
        "  {} x {} nodes, nnz = {}, in {:.2} ms  ->  {:.0} rows/sec\n",
        n,
        n,
        adj.nnz(),
        dt * 1e3,
        n as f64 / dt
    );
}

fn bench_bitsdr() {
    let mut rng = seeded_rng(3);
    let n = 4096;
    let dim = 256;
    // ~2% dense random binary SDRs
    let dense = Array2::from_shape_fn((n, dim), |_| {
        if rng.gen_range(0.0_f32..1.0) < 0.02 {
            1.0
        } else {
            0.0
        }
    });
    let packed = pack_batch(&dense.view());

    let t0 = Instant::now();
    let mut acc = 0u64;
    for i in 0..n {
        for j in (i + 1)..n.min(i + 256) {
            acc += packed[i].overlap(&packed[j]) as u64;
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    let pairs = (0..n).map(|i| n.min(i + 256) - i - 1).sum::<usize>();
    println!("[BitSdr overlap]  256-bit packed SDR popcount similarity");
    println!(
        "  {} pairs in {:.2} ms  ->  {:.1} M pairs/sec (checksum {})\n",
        pairs,
        dt * 1e3,
        pairs as f64 / dt / 1e6,
        acc
    );
}

fn bench_pipeline() {
    let shapes = [(8usize, 64usize), (32, 128), (64, 256)];
    for (batch, seq) in shapes {
        run_pipeline_case(batch, seq);
    }
}

fn run_pipeline_case(batch: usize, seq: usize) {
    let mut rng = seeded_rng(4);
    let ids = Array2::from_shape_fn((batch, seq), |_| rng.gen_range(0.0_f32..30000.0).floor());
    let mask = Array2::<f32>::ones((batch, seq));
    let nodes: Vec<usize> = (0..batch).map(|b| b % seq).collect();

    // Sparse adjacency over the sequence graph (ring + skip connections),
    // representative of the thresholded cosine adjacency density.
    let mut dense = Array2::<f32>::zeros((seq, seq));
    for i in 0..seq {
        for d in 1..=8usize {
            dense[(i, (i + d) % seq)] = 1.0;
            dense[(i, (i + seq - d) % seq)] = 1.0;
        }
    }
    let adj = CsrMatrix::from_dense(&dense.view());

    println!(
        "[Pipeline]  batch={batch} seq={seq}  ({} tokens/pass, adjacency nnz={})",
        batch * seq,
        adj.nnz()
    );

    let exact_out = run_config(
        "exact (dense)",
        ModelConfig::default(),
        batch,
        seq,
        &ids,
        &mask,
        &adj,
        &nodes,
    );
    let sparse_out = run_config(
        "sparse-exact (event-driven, dense recurrent, chunked)",
        ModelConfig {
            sparsity: SparsityConfig::aggressive(),
            ..Default::default()
        },
        batch,
        seq,
        &ids,
        &mask,
        &adj,
        &nodes,
    );
    let extreme_out = run_config(
        "sparse-extreme (event-driven, rec keep 25% CSR, chunked)",
        ModelConfig {
            sparsity: SparsityConfig::extreme(),
            ..Default::default()
        },
        batch,
        seq,
        &ids,
        &mask,
        &adj,
        &nodes,
    );

    let cosine = |a: &Array2<f32>, b: &Array2<f32>| {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        dot / (na * nb).max(1e-12)
    };
    println!(
        "  fidelity vs exact: sparse-exact cosine = {:.4}, sparse-extreme cosine = {:.4}\n",
        cosine(&exact_out, &sparse_out),
        cosine(&exact_out, &extreme_out)
    );
}

#[allow(clippy::too_many_arguments)]
fn run_config(
    label: &str,
    cfg: ModelConfig,
    batch: usize,
    seq: usize,
    ids: &Array2<f32>,
    mask: &Array2<f32>,
    adj: &CsrMatrix,
    nodes: &[usize],
) -> Array2<f32> {
    let mut model = SnnModel::new(cfg);
    // warmup
    let _ = model.forward(&ids.view(), &mask.view(), Some(adj), nodes, false);

    let iters = 5;
    let mut sum = LayerTimings::default();
    let mut last = Array2::<f32>::zeros((0, 0));
    for _ in 0..iters {
        let (out, t) = model.forward_timed(&ids.view(), &mask.view(), Some(adj), nodes, false);
        std::hint::black_box(&out);
        last = out;
        sum.input_projection += t.input_projection;
        sum.snn += t.snn;
        sum.temporal_attention += t.temporal_attention;
        sum.gat += t.gat;
        sum.htm += t.htm;
        sum.head += t.head;
        sum.total += t.total;
    }
    let inv = 1.0 / iters as f64;
    let tokens = (batch * seq) as f64;

    println!("  --- {label} ---");
    let row = |name: &str, secs: f64| {
        println!(
            "  {:<22} {:>9.3} ms   {:>12.0} tokens/sec",
            name,
            secs * inv * 1e3,
            tokens / (secs * inv)
        );
    };
    row("input projection", sum.input_projection);
    row("SNN (10 timesteps)", sum.snn);
    row("temporal attention", sum.temporal_attention);
    row("GATv2 + skip + mask", sum.gat);
    row("HTM + skip", sum.htm);
    row("gather + MLP head", sum.head);
    row("TOTAL", sum.total);
    last
}
