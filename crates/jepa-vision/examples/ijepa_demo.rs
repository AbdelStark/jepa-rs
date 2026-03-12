//! I-JEPA demonstration: end-to-end forward pass pipeline.
//!
//! This example demonstrates the full I-JEPA architecture:
//! 1. Create a ViT encoder pair (context + target)
//! 2. Create a transformer predictor
//! 3. Generate block masks
//! 4. Encode input images
//! 5. Predict target representations from context
//! 6. Compute energy (prediction loss)
//! 7. Compute collapse prevention loss (VICReg)
//! 8. Report training metrics
//!
//! Run with: `cargo run --example ijepa_demo`

use burn::prelude::*;
use burn::tensor::ElementConversion;
use burn_ndarray::NdArray;
use rand::SeedableRng;

use jepa_core::collapse::VICReg;
use jepa_core::energy::{EnergyFn, L2Energy};
use jepa_core::masking::{BlockMasking, MaskingStrategy};
use jepa_core::types::InputShape;
use jepa_core::Predictor;
use jepa_vision::image::{IJepaConfig, TransformerPredictorConfig};
use jepa_vision::vit::VitConfig;

type B = NdArray<f32>;

fn main() {
    println!("=== I-JEPA Demo ===\n");

    let device = burn_ndarray::NdArrayDevice::Cpu;

    // --- 1. Model Configuration ---
    let encoder_config = VitConfig {
        in_channels: 3,
        image_height: 32,
        image_width: 32,
        patch_size: (4, 4),
        embed_dim: 64,
        num_layers: 2,
        num_heads: 4,
        mlp_dim: 256,
        dropout: 0.0,
    };

    let predictor_config = TransformerPredictorConfig {
        encoder_embed_dim: encoder_config.embed_dim,
        predictor_embed_dim: 32,
        num_layers: 1,
        num_heads: 4,
        max_target_len: 64,
    };

    let config = IJepaConfig {
        encoder: encoder_config.clone(),
        predictor: predictor_config,
    };

    println!("Model config:");
    println!(
        "  Encoder: embed_dim={}, layers={}, heads={}",
        encoder_config.embed_dim, encoder_config.num_layers, encoder_config.num_heads
    );
    println!(
        "  Image: {}x{}, patch: {}x{}",
        encoder_config.image_height,
        encoder_config.image_width,
        encoder_config.patch_size.0,
        encoder_config.patch_size.1
    );
    let grid_h = encoder_config.image_height / encoder_config.patch_size.0;
    let grid_w = encoder_config.image_width / encoder_config.patch_size.1;
    let num_patches = grid_h * grid_w;
    println!(
        "  Patch grid: {}x{} = {} patches",
        grid_h, grid_w, num_patches
    );
    println!();

    // --- 2. Initialize Model ---
    let model = config.init::<B>(&device);
    println!("Model initialized (context encoder + target encoder + predictor)");
    println!();

    // --- 3. Create Synthetic Input ---
    let batch_size = 4;
    let images: Tensor<B, 4> = Tensor::random(
        [batch_size, 3, 32, 32],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    println!(
        "Input: batch={}, channels=3, height=32, width=32",
        batch_size
    );
    println!();

    // --- 4. Masking ---
    let masking = BlockMasking {
        num_targets: 4,
        target_scale: (0.15, 0.2),
        target_aspect_ratio: (0.75, 1.5),
    };
    let input_shape = InputShape::Image {
        height: grid_h,
        width: grid_w,
    };
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let mask = masking.generate_mask(&input_shape, &mut rng);

    println!("Mask generated:");
    println!("  Context (visible) tokens: {}", mask.context_indices.len());
    println!("  Target (hidden) tokens:   {}", mask.target_indices.len());
    println!(
        "  Mask ratio:               {:.1}%",
        mask.mask_ratio() * 100.0
    );
    assert!(mask.validate().is_ok(), "mask validation failed");
    println!();

    // --- 5. Encoding ---
    let context_repr = model.context_encoder.forward(&images);
    let target_repr = model.target_encoder.forward(&images);

    println!(
        "Context encoder output: [{}, {}, {}]",
        context_repr.batch_size(),
        context_repr.seq_len(),
        context_repr.embed_dim()
    );
    println!(
        "Target encoder output:  [{}, {}, {}]",
        target_repr.batch_size(),
        target_repr.seq_len(),
        target_repr.embed_dim()
    );
    println!();

    // --- 6. Prediction ---
    let num_targets = mask.target_indices.len();
    let target_positions: Tensor<B, 2> = Tensor::zeros([batch_size, num_targets], &device);
    let predicted = model
        .predictor
        .predict(&context_repr, &target_positions, None);

    println!(
        "Predicted target: [{}, {}, {}]",
        predicted.batch_size(),
        predicted.seq_len(),
        predicted.embed_dim()
    );
    println!();

    // --- 7. Energy Computation ---
    let target_slice = jepa_core::types::Representation::new(target_repr.embeddings.slice([
        0..batch_size,
        0..num_targets,
        0..encoder_config.embed_dim,
    ]));

    let energy = L2Energy.compute(&predicted, &target_slice);
    let energy_val: f32 = energy.value.into_scalar().elem();
    println!("L2 Energy (prediction loss): {:.6}", energy_val);

    // Self-energy should be ~0
    let self_energy = L2Energy.compute(&predicted, &predicted);
    let self_val: f32 = self_energy.value.into_scalar().elem();
    println!(
        "Self-energy (sanity check):  {:.6} (should be ~0)",
        self_val
    );
    println!();

    // --- 8. Collapse Prevention ---
    let vicreg = VICReg::default();
    let pred_flat = predicted
        .embeddings
        .clone()
        .reshape([batch_size * num_targets, encoder_config.embed_dim]);
    let target_flat = target_slice
        .embeddings
        .clone()
        .reshape([batch_size * num_targets, encoder_config.embed_dim]);
    let vicreg_loss = vicreg.compute(&pred_flat, &target_flat);

    let inv: f32 = vicreg_loss.invariance.into_scalar().elem();
    let var: f32 = vicreg_loss.variance.into_scalar().elem();
    let cov: f32 = vicreg_loss.covariance.into_scalar().elem();
    let total = inv + var + cov;

    println!("VICReg Loss:");
    println!("  Invariance:  {:.4}", inv);
    println!("  Variance:    {:.4}", var);
    println!("  Covariance:  {:.4}", cov);
    println!("  Total:       {:.4}", total);
    println!();

    // --- 9. Combined Training Loss ---
    let reg_weight = 1.0;
    let total_loss = energy_val + reg_weight as f32 * total;
    println!(
        "Total training loss: {:.4} (energy) + {:.1} * {:.4} (reg) = {:.4}",
        energy_val, reg_weight, total, total_loss
    );
    println!();

    // --- 10. EMA Update Demo ---
    let ema = jepa_core::ema::Ema::with_cosine_schedule(0.996, 100_000);
    println!("EMA Schedule (cosine, 0.996 -> 1.0):");
    for &step in &[0, 10_000, 50_000, 99_999] {
        println!(
            "  Step {:>6}: momentum = {:.6}",
            step,
            ema.get_momentum(step)
        );
    }
    println!();

    println!("=== Demo Complete ===");
}
