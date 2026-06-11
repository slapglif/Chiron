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

## Sparsity engine

The dominant intermediate signal is the SNN spike cube `(N, T, 128)`.
Three measured facts drive the sparse execution paths (`SparsityConfig`):

1. **Spikes are binary and LayerNorm of a binary vector takes exactly two
   values per row**, fully determined by the spike count (mean `m = k/dim`,
   `var = m(1-m)`). The cube is therefore carried as bit-packed `u64` masks
   (16 B per row instead of 512 B, 32x), and every downstream
   `Linear(LayerNorm(spikes))` collapses *exactly* into a sparse accumulation
   of pre-scaled weight rows plus a rank-1 correction derived from popcount
   (`SpikeTrain`, `SnnLayer::forward_events`,
   `TemporalAttention::forward_events`). Measured spike density ramps
   4.7% (t=0) to ~47% (t=9) as the ALIF adaptation engages.
2. **The HTM inhibition is degenerate at config dimensions**: it selects
   `min(10, 8) = 8` winners out of 8 minicolumns, so the active mask is
   constant all-ones and at inference the entire pooler collapses to a
   precomputed constant vector + residual projection (detected via
   `is_inhibition_degenerate()`, ~2.5x on the HTM stage; training keeps the
   full plasticity path). Non-degenerate configs get O(k·sdr) sparse output
   projection and CSR connected-overlap instead.
3. **Chunked fused execution**: input-proj -> SNN -> temporal attention run
   chunk-by-chunk (default 2048 rows), bounding the live working set instead
   of streaming the full float cube through DRAM twice.

Presets: `SparsityConfig::exact()` (dense reference),
`aggressive()` (everything above — **no approximation**, output cosine
1.0000 vs exact), `extreme()` (adds 25%-kept CSR recurrent weights — the
one lossy knob, cosine ~0.86).

**Negative result, kept for honesty:** magnitude-pruning the recurrent
membrane GEMM to CSR does *not* pay at these dimensions. At 50% kept
weights the gather-bound CSR matvec loses outright to the SIMD dense GEMM,
and even at 25% it merely breaks even while costing 0.86 fidelity — so
`aggressive()` keeps the recurrent GEMM dense. The recurrence is the
irreducible core: 10 sequential timesteps of `mem @ W_rec` on a dense
membrane state.

## Measured throughput

4 threads, `--release` (lto=fat, codegen-units=1), config.json dimensions:

```
[SDR generation]  300d -> 64d -> 256d binary, sparsity 0.002
  8192 embeddings in 12.50 ms  ->  655,547 embeddings/sec

[Adjacency build] cosine kNN, threshold 0.5, top_k 50
  1024 nodes, nnz = 31,744, in 9.68 ms  ->  105,825 rows/sec

[BitSdr overlap]  256-bit packed SDR popcount similarity
  226.5 M pairs/sec

[Pipeline]  batch=32 seq=128         exact      sparse-exact   speedup
  input projection                  1.88 ms        2.15 ms      0.9x
  SNN (10 timesteps)               92.40 ms       58.12 ms      1.6x
  temporal attention               50.07 ms       10.79 ms      4.6x
  GATv2 + skip + mask              25.62 ms       25.75 ms      1.0x
  HTM + skip                       13.65 ms       11.32 ms      1.2x
  gather + MLP head                 0.44 ms        0.42 ms      1.0x
  TOTAL                           202.02 ms      108.85 ms      1.9x
  output cosine vs exact:                          1.0000

[Pipeline]  batch=64 seq=256         exact      sparse-exact   speedup
  SNN (10 timesteps)              420.51 ms      248.39 ms      1.7x
  temporal attention              279.48 ms       44.46 ms      6.3x
  TOTAL                           878.86 ms      430.74 ms      2.0x
  output cosine vs exact:                          1.0000
```

After sparsification the irreducible bottleneck is the SNN's recurrent
membrane GEMM: 10 sequential timesteps of `mem @ W_rec` over a dense float
membrane state — it cannot be event-driven (membranes are not binary) and
CSR pruning measurably loses to the dense GEMM (see above). Everything
around it has been sparsified exactly. Reproduce with:

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
