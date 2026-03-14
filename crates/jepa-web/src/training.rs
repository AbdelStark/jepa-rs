//! Browser-adapted training loop for I-JEPA.
//!
//! Instead of a blocking loop, this module exposes a single `train_step`
//! function that executes one gradient update. The JavaScript layer calls
//! this repeatedly via `requestAnimationFrame` or `setInterval` to yield
//! back to the browser event loop between steps.

use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;

use jepa_core::energy::L2Energy;
use jepa_core::masking::MaskingStrategy;
use jepa_core::VICReg;
use jepa_train::schedule::LrSchedule;

use crate::backend::{CpuBackend, CPU_DEVICE};
use crate::state::{StepMetrics, TrainingState};

/// Execute a single training step and return metrics.
///
/// This is the core function called from JavaScript on each animation frame.
/// It generates synthetic data, runs the I-JEPA forward pass, computes
/// gradients, steps the optimizer, and updates the target encoder via EMA.
pub fn train_step(state: &mut TrainingState) -> StepMetrics {
    let step = state.current_step;
    let lr = state.lr_schedule.get_lr(step);
    let ema_momentum = state.ema.get_momentum(step);

    // Generate synthetic random batch.
    let batch_size = state.config.batch_size;
    let in_channels = state.vit_config.in_channels;
    let height = state.vit_config.image_height;
    let width = state.vit_config.image_width;
    let input: Tensor<CpuBackend, 4> = Tensor::random(
        [batch_size, in_channels, height, width],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &CPU_DEVICE,
    );

    // Generate mask.
    let mask = state
        .masking
        .generate_mask(&state.input_shape, &mut state.rng);

    // Forward pass (strict masking).
    let energy_fn = L2Energy;
    let regularizer = VICReg::default();
    let output = state.model.forward_step_strict(
        &input,
        mask,
        &energy_fn,
        &regularizer,
        state.config.reg_weight,
    );

    // Extract scalar metrics before backward pass consumes the graph.
    let energy_val: f64 = f64::from(output.energy.value.clone().into_scalar());
    let reg_val: f64 = f64::from(output.regularization.clone().into_scalar());
    let total_val: f64 = f64::from(output.total_loss.clone().into_scalar());

    // Backward pass and optimizer step.
    let grads = GradientsParams::from_grads(output.total_loss.backward(), &state.model);
    state.model = state.optimizer.step(lr, state.model.clone(), grads);

    // EMA update of target encoder.
    state.model.target_encoder = state
        .model
        .target_encoder
        .clone()
        .ema_update_from(&state.model.context_encoder, &state.ema, step)
        .no_grad();

    state.current_step += 1;

    StepMetrics {
        step,
        energy: energy_val,
        regularization: reg_val,
        total_loss: total_val,
        learning_rate: lr,
        ema_momentum,
    }
}
