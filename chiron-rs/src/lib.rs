//! chiron-core: high-performance Rust implementation of the Chiron semantic
//! folding pipeline core layers.
//!
//! Signal flow (mirrors `chiron/layers/snn/model.py::SNNModel.forward`):
//!
//! ```text
//! token ids (B, S)
//!   -> input projection Linear(1 -> 256) + LayerNorm
//!   -> SNNLayer        fc(256->64) -> PLIF(64) -> LN -> fc(64->128) -> PLIF(128) -> LN
//!                      repeated for T=10 timesteps        => (B, S, T, 128)
//!   -> TemporalAttention over timesteps                   => (B, S, 128)
//!   -> GATv2 (128 -> 64, 2 heads, adjacency-masked) + SNN skip
//!   -> attention-mask multiply
//!   -> HTM spatial pooler (pad 64->128, 8 minicolumns -> 256) + GAT skip
//!   -> node gather -> MLP head (256 -> 128 -> 256)        => (B, 256)
//! ```

pub mod nn;
pub mod sparse;
pub mod sdr;
pub mod snn;
pub mod gat;
pub mod htm;
pub mod model;
