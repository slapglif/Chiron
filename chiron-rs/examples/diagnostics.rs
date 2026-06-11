//! Diagnostic: measure spike density inside the SNN layer and verify the
//! HTM inhibition degeneracy at config.json dimensions.
//!
//! Run: cargo run --release --example diagnostics

use chiron_core::htm::{HtmParams, HtmSpatialPooler};
use chiron_core::nn::{seeded_rng, LayerNorm, Linear};
use chiron_core::snn::SnnLayer;
use ndarray::{Array2, Axis};
use rand::Rng;

fn main() {
    spike_density();
    htm_degeneracy();
}

fn spike_density() {
    let mut rng = seeded_rng(42);
    let n = 4096;

    // Replicate the pipeline input stage: scalar id -> Linear(1,256) -> LN.
    let ids = Array2::from_shape_fn((n, 1), |_| rng.gen_range(0.0_f32..30000.0).floor());
    let proj = Linear::new(1, 256, true, &mut rng);
    let ln = LayerNorm::new(256);
    let mut x = proj.forward(&ids.view());
    ln.forward_inplace(x.view_mut());

    let snn = SnnLayer::new(256, 64, 128, 10, &mut rng);
    let out = snn.forward(&x.view()); // (n, 10, 128) -- post-LN spike values

    // Post-LN spikes: per (n,t) row the spiking entries are the values
    // strictly above the row mean (binary input affinely mapped with a
    // positive scale). All-equal rows mean zero spikes.
    println!("SNN output-layer (128 neurons) spike density per timestep:");
    for t in 0..10 {
        let mut active = 0usize;
        let mut total = 0usize;
        for nrow in 0..n {
            let row = out.index_axis(Axis(0), nrow);
            let row_t = row.index_axis(Axis(0), t);
            let mean = row_t.sum() / row_t.len() as f32;
            let mx = row_t.iter().cloned().fold(f32::MIN, f32::max);
            let mn = row_t.iter().cloned().fold(f32::MAX, f32::min);
            if (mx - mn).abs() > 1e-6 {
                active += row_t.iter().filter(|&&v| v > mean).count();
            }
            total += row_t.len();
        }
        println!("  t={:2}: {:.2}%", t, 100.0 * active as f64 / total as f64);
    }
}

fn htm_degeneracy() {
    let params = HtmParams::default();
    let mut sp = HtmSpatialPooler::new(&params);
    let mut rng = seeded_rng(7);
    let x = Array2::from_shape_fn((256, 128), |_| rng.gen_range(-1.0_f32..1.0));
    let out = sp.forward(&x.view(), false);
    let (rows, cols) = out.dim();
    let all_ones = out.iter().all(|&v| v == 1.0);
    println!("\nHTM spatial pooler at config dims: {} minicolumns, output {}x{}", cols, rows, cols);
    println!("  active mask is all-ones (inhibition degenerate): {}", all_ones);
}
