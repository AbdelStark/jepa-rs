//! I-JEPA training loop simulation.
//!
//! Demonstrates the full JEPA training pipeline across crates:
//! - jepa-core: Masking, energy, EMA, VICReg, config
//! - jepa-vision: ViT encoder pair + transformer predictor
//! - jepa-train: Learning rate schedule, metrics, checkpoint metadata
//!
//! This is a simulation (not real training with backprop) that exercises
//! the full forward pass and metrics tracking over multiple steps.
//!
//! Run with: `cargo run --example ijepa_train_loop -p jepa-vision`

use burn::prelude::*;
use burn::tensor::ElementConversion;
use burn_ndarray::NdArray;
use rand::SeedableRng;

use jepa_core::collapse::VICReg;
use jepa_core::ema::Ema;
use jepa_core::energy::{EnergyFn, L2Energy};
use jepa_core::masking::{BlockMasking, MaskingStrategy};
use jepa_core::types::InputShape;
use jepa_core::Predictor;
use jepa_train::schedule::{LrSchedule, WarmupCosineSchedule};
use jepa_train::{CheckpointMeta, TrainConfig, TrainMetrics};
use jepa_vision::image::{IJepaConfig, TransformerPredictorConfig};
use jepa_vision::vit::VitConfig;

type B = NdArray<f32>;

fn main() {
    println!("=== I-JEPA Training Loop Simulation ===\n");

    let device = burn_ndarray::NdArrayDevice::Cpu;

    // --- Configuration ---
    let train_config = TrainConfig {
        total_steps: 50,
        warmup_steps: 10,
        peak_lr: 1.5e-4,
        regularization_weight: 1.0,
        ema_momentum: 0.996,
        batch_size: 2,
        log_interval: 10,
        checkpoint_interval: 25,
    };
    assert!(train_config.validate().is_ok());

    let encoder_config = VitConfig {
        in_channels: 3,
        image_height: 16,
        image_width: 16,
        patch_size: (4, 4),
        embed_dim: 32,
        num_layers: 1,
        num_heads: 4,
        mlp_dim: 128,
        dropout: 0.0,
    };

    let predictor_config = TransformerPredictorConfig {
        encoder_embed_dim: encoder_config.embed_dim,
        predictor_embed_dim: 16,
        num_layers: 1,
        num_heads: 4,
        max_target_len: 16,
    };

    let model_config = IJepaConfig {
        encoder: encoder_config.clone(),
        predictor: predictor_config,
    };

    let grid_h = encoder_config.image_height / encoder_config.patch_size.0;
    let grid_w = encoder_config.image_width / encoder_config.patch_size.1;

    println!("Config:");
    println!(
        "  Steps: {} (warmup: {})",
        train_config.total_steps, train_config.warmup_steps
    );
    println!("  Peak LR: {}", train_config.peak_lr);
    println!("  EMA momentum: {}", train_config.ema_momentum);
    println!("  Image: {}x{}, patch grid: {}x{}", 16, 16, grid_h, grid_w);
    println!();

    // --- Initialize ---
    let model = model_config.init::<B>(&device);
    let masking = BlockMasking {
        num_targets: 2,
        target_scale: (0.15, 0.3),
        target_aspect_ratio: (0.75, 1.5),
    };
    let input_shape = InputShape::Image {
        height: grid_h,
        width: grid_w,
    };
    let lr_schedule = WarmupCosineSchedule::new(
        train_config.peak_lr,
        train_config.warmup_steps,
        train_config.total_steps,
    );
    let ema = Ema::with_cosine_schedule(train_config.ema_momentum, train_config.total_steps);
    let vicreg = VICReg::default();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let mut metrics = TrainMetrics::default();
    let mut checkpoint = CheckpointMeta::new(train_config.total_steps);

    println!("--- Training Loop ---\n");

    for step in 0..train_config.total_steps {
        // Generate synthetic batch
        let images: Tensor<B, 4> = Tensor::random(
            [train_config.batch_size, 3, 16, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Forward pass
        let mask = masking.generate_mask(&input_shape, &mut rng);
        let context_repr = model.context_encoder.forward(&images);
        let target_repr = model.target_encoder.forward(&images);

        let num_targets = mask.target_indices.len();
        let target_positions: Tensor<B, 2> =
            Tensor::zeros([train_config.batch_size, num_targets], &device);
        let predicted = model
            .predictor
            .predict(&context_repr, &target_positions, None);

        // Target gathering
        let target_slice = target_repr.gather(&mask.target_indices);

        // Energy
        let energy = L2Energy.compute(&predicted, &target_slice);
        let energy_val: f32 = energy.value.into_scalar().elem();

        // Regularization
        let embed_dim = predicted.embed_dim();
        let batch = predicted.batch_size();
        let pred_flat = predicted
            .embeddings
            .clone()
            .reshape([batch * num_targets, embed_dim]);
        let tgt_flat = target_slice
            .embeddings
            .clone()
            .reshape([batch * num_targets, embed_dim]);
        let vicreg_loss = vicreg.compute(&pred_flat, &tgt_flat);
        let reg_val: f32 = vicreg_loss.total().into_scalar().elem();
        let total_loss = energy_val + train_config.regularization_weight as f32 * reg_val;

        // Track metrics
        metrics.record(energy_val as f64, reg_val as f64, total_loss as f64);

        // Schedule values
        let lr = lr_schedule.get_lr(step);
        let momentum = ema.get_momentum(step);

        // Log interval
        if (step + 1) % train_config.log_interval == 0 || step == 0 {
            let (avg_e, avg_r, avg_t) = metrics.take_averages();
            println!(
                "Step {:>3}/{}: loss={:.4} (energy={:.4} reg={:.4}) lr={:.2e} mom={:.6}",
                step + 1,
                train_config.total_steps,
                avg_t,
                avg_e,
                avg_r,
                lr,
                momentum,
            );
        }

        // Checkpoint interval
        if (step + 1) % train_config.checkpoint_interval == 0 {
            checkpoint.step = step + 1;
            checkpoint.learning_rate = lr;
            checkpoint.ema_momentum = momentum;
            checkpoint.last_loss = Some(total_loss as f64);
            println!(
                "  [Checkpoint] step={}, progress={:.0}%, loss={:.4}",
                checkpoint.step,
                checkpoint.progress() * 100.0,
                total_loss,
            );
        }
    }

    // Final checkpoint
    checkpoint.step = train_config.total_steps;
    println!();
    println!("--- Training Complete ---");
    println!(
        "Final checkpoint: step={}, progress={:.0}%, complete={}",
        checkpoint.step,
        checkpoint.progress() * 100.0,
        checkpoint.is_complete(),
    );
    println!();
    println!("=== Simulation Done ===");
}
