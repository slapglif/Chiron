# chiron-core

High-performance Rust implementation of the Chiron semantic-folding pipeline
core (SDR / SNN / GAT / HTM), ported from the PyTorch reference in
`chiron/layers/` with the inference signal path reproduced layer-for-layer.

## Signal flow

Mirrors `chiron/layers/snn/model.py::SNNModel.forward` at the dimensions from
`chiron/config.json`:

```
token ids (B, S)
  │ Linear(1 -> 256) + LayerNorm                    learned input embedding
  ▼
SNNLayer                                            (B·S, 256)
  │ fc1(256 -> 64) — constant injected current, computed once
  │ ×10 timesteps:
  │   PLIF(64, recurrent)  -> LayerNorm
  │   fc2(64 -> 128)
  │   PLIF(128, recurrent) -> LayerNorm
  ▼                                                 (B·S, 10, 128)
TemporalAttention (mean-query over timesteps)       (B·S, 128)
  ▼
GATv2 (128 -> 64, 2 heads, adjacency-masked)        (B, S, 64)
  + skip: Linear(128 -> 64) from SNN output, LayerNorm
  × attention mask
  ▼
HTM spatial pooler                                  (B·S, 256)
  zero-pad 64 -> 128, input_proj, overlap GEMM,
  boost, top-8-of-8 inhibition, fc(8 -> 256),
  residual, LayerNorm
  + skip: Linear(64 -> 256) from GAT output, LayerNorm
  ▼
node gather -> MLP head (256 -> 128 -> 256)         (B, 256)
```

Plus the standalone encoding path: `SdrGenerator`
(Word2Vec 300-d → Linear → LayerNorm → FlyHash expansion → k-WTA → binary SDR)
and `CsrMatrix::cosine_adjacency` (cosine similarity → threshold → top-k CSR).

## Optimizations over the Python reference

| Site | Python reference | Rust implementation |
|---|---|---|
| GAT attention | Materialises full S×S logits, then masks with `dtype.min` | Sparse path scores **only CSR edges**: O(nnz·d_k) instead of O(S²·d_k); per-thread scratch, zero allocations in the edge loop |
| GAT projections | Per-head einsum over a (H, F, d_k) tensor | W_q/W_k/W_v fused to single (F, H·d_k) matrices — 3 GEMMs total |
| PLIF neurons | Recomputes `exp/sigmoid/softplus` decay constants **every timestep** | `beta_mem`, `beta_syn`, `V_th` precomputed once at construction |
| PLIF state update | Separate tensor ops (syn, mem, spike, reset, adapt) | Single fused rayon pass per row; only the recurrent GEMM is separate |
| SNN input current | `fc1` inside the timestep loop graph | Hoisted: computed once, injected at all 10 steps |
| HTM overlap | Rebuilds connected mask each forward (sigmoid + STE), then matmul | Connected matrix cached **transposed** (input, C); rebuilt only when permanences change; overlap is one GEMM |
| Top-k (kWTA, HTM inhibition) | `torch.topk` (sort-based on CPU) | `select_nth_unstable` O(n) threshold |
| Adjacency slicing | Sparse → dense → slice | CSR principal-submatrix extraction, never densified |
| SDR storage/similarity | Dense float arrays | Bit-packed u64 words, popcount overlap / Jaccard / Hamming |

Everything is `f32`, GEMM via `matrixmultiply` (blocked, SIMD), rayon
row-parallelism, and fully deterministic for a given config seed.

Documented behavioural deviation: GAT rows with **no** neighbours produce zero
attention output (pure residual pathway). The Python masks with `dtype.min`,
which degenerates to *uniform attention over all nodes* for isolated rows — an
artifact of finite masking, not intent.

## Measured throughput

4 threads, `--release` (lto=fat, codegen-units=1), config.json dimensions:

```
[SDR generation]  300d -> 64d -> 256d binary, sparsity 0.002
  8192 embeddings in 12.50 ms  ->  655,547 embeddings/sec

[Adjacency build] cosine kNN, threshold 0.5, top_k 50
  1024 nodes, nnz = 31,744, in 9.68 ms  ->  105,825 rows/sec

[BitSdr overlap]  256-bit packed SDR popcount similarity
  226.5 M pairs/sec

[Pipeline]  batch=32 seq=128 (4096 tokens/pass)
  input projection           1.54 ms     2,666,762 tokens/sec
  SNN (10 timesteps)        89.32 ms        45,858 tokens/sec
  temporal attention        38.10 ms       107,496 tokens/sec
  GATv2 + skip + mask       16.71 ms       245,151 tokens/sec
  HTM + skip                12.57 ms       325,942 tokens/sec
  gather + MLP head          0.35 ms    11,765,064 tokens/sec
  TOTAL                    158.58 ms        25,829 tokens/sec
```

The SNN dominates (~56% of wall time): its cost is inherent — 10 sequential
timesteps × (recurrent GEMM + projection GEMM + LayerNorm) that cannot be
batched across time because of the membrane-state dependency. Temporal
attention is the next-largest term and is GEMM-bound (key projection over
N·T rows). Reproduce with:

```bash
cd chiron-rs
cargo run --release --bin bench
```

## Layout

```
src/nn.rs      Linear / LayerNorm / softmax / GELU / top-k / seeded init
src/sparse.rs  CSR matrix, cosine kNN adjacency, principal submatrix
src/sdr.rs     FlyHash projection, k-WTA, SdrGenerator, bit-packed SDRs
src/snn.rs     PLIF/ALIF neurons, SNNLayer, TemporalAttention, MLPHead
src/gat.rs     GATv2 (edge-only sparse path + dense path)
src/htm.rs     HTM spatial pooler + HtmModel (plasticity incl.)
src/model.rs   SnnModel: full pipeline wiring + per-layer timing
src/bin/bench.rs  throughput benchmark
```

```bash
cargo test          # 42 unit + integration tests
cargo clippy        # zero warnings
```

## Scope note

This crate implements the **forward/inference path** plus HTM's local
(non-gradient) learning rules — Hebbian permanence updates, duty-cycle EMA,
and boosting. Gradient-based training (surrogate spike gradients, STE) stays
in the PyTorch reference, which remains the training harness.
