#![recursion_limit = "256"]
//! # jepa-web
//!
//! WebGPU browser demo for jepa-rs.
//!
//! Provides a complete in-browser JEPA training and inference experience
//! using `burn-wgpu` (WebGPU) with CPU fallback via `burn-ndarray`. All
//! computation runs client-side with zero server dependencies.
//!
//! ## Architecture
//!
//! - [`backend`]: Backend type aliases (`Autodiff<Wgpu>`, `Autodiff<NdArray>`)
//! - [`state`]: Training session state management
//! - [`training`]: Single-step training loop called from JS
//! - [`inference`]: Encoder forward pass and embedding visualization
//!
//! ## Entry Points
//!
//! The `#[wasm_bindgen]` functions in this module are the JS-callable API:
//!
//! | Function | Purpose |
//! |----------|---------|
//! | `init_demo` | Set up panic hook and console logging |
//! | `create_training_session` | Initialize model and optimizer from config |
//! | `training_step` | Run one gradient step, return metrics as JSON |
//! | `run_inference_on_pattern` | Run encoder on a named demo pattern |
//! | `run_inference_on_data` | Run encoder on user-provided pixel data |
//! | `get_model_info` | Return model architecture summary |

pub mod backend;
pub mod inference;
pub mod state;
pub mod training;

use std::cell::RefCell;

use wasm_bindgen::prelude::*;

use crate::backend::{CpuBackend, CPU_DEVICE};
use crate::state::{TrainingConfig, TrainingState};

thread_local! {
    /// Global training state held in a thread-local `RefCell`.
    ///
    /// WASM is single-threaded so `thread_local!` + `RefCell` is safe and avoids
    /// the `static_mut_refs` lint that fires on `static mut`.
    static TRAINING_STATE: RefCell<Option<TrainingState>> = const { RefCell::new(None) };
}

/// Helper: run a closure with a mutable reference to the training state.
fn with_state_mut<T>(f: impl FnOnce(&mut TrainingState) -> T) -> Result<T, String> {
    TRAINING_STATE.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let state = borrow.as_mut().ok_or_else(|| {
            "no training session — call create_training_session first".to_string()
        })?;
        Ok(f(state))
    })
}

/// Helper: run a closure with a shared reference to the training state.
fn with_state<T>(f: impl FnOnce(&TrainingState) -> T) -> Result<T, String> {
    TRAINING_STATE.with(|cell| {
        let borrow = cell.borrow();
        let state = borrow.as_ref().ok_or_else(|| {
            "no training session — call create_training_session first".to_string()
        })?;
        Ok(f(state))
    })
}

/// Initialize the demo runtime.
///
/// Sets up the panic hook for better error messages in the browser console.
/// Call this once before any other functions.
#[wasm_bindgen]
pub fn init_demo() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"[jepa-web] Demo initialized".into());
}

/// Create a new training session with the given configuration.
///
/// Accepts a JSON-serialized `TrainingConfig`. Returns a JSON summary of the
/// initialized model.
///
/// # Errors
///
/// Returns an error string if the config JSON is invalid.
#[wasm_bindgen]
pub fn create_training_session(config_json: &str) -> Result<String, String> {
    let config: TrainingConfig =
        serde_json::from_str(config_json).map_err(|e| format!("invalid config: {e}"))?;

    let training_state = state::init_training_state(config, &CPU_DEVICE);
    let info = model_info_from_state(&training_state);

    TRAINING_STATE.with(|cell| {
        *cell.borrow_mut() = Some(training_state);
    });

    serde_json::to_string(&info).map_err(|e| format!("serialization error: {e}"))
}

/// Execute a single training step.
///
/// Returns a JSON-serialized `StepMetrics` with loss values and schedule info.
///
/// # Errors
///
/// Returns an error if no training session has been created.
#[wasm_bindgen]
pub fn training_step() -> Result<String, String> {
    let metrics = with_state_mut(training::train_step)?;
    serde_json::to_string(&metrics).map_err(|e| format!("serialization error: {e}"))
}

/// Run inference on a named demo pattern.
///
/// Available pattern names: "gradient", "checkerboard", "rings", "noise".
///
/// Returns a JSON-serialized `InferenceResult` with embedding statistics
/// and per-patch norms.
///
/// # Errors
///
/// Returns an error if the pattern name is unknown or no session exists.
#[wasm_bindgen]
pub fn run_inference_on_pattern(pattern_name: &str) -> Result<String, String> {
    with_state(|state| {
        let patterns = inference::demo_patterns::<CpuBackend>(&state.vit_config, &CPU_DEVICE);

        let (_name, input) = patterns
            .into_iter()
            .find(|(name, _)| *name == pattern_name)
            .ok_or_else(|| {
                format!(
                    "unknown pattern '{pattern_name}'; available: gradient, checkerboard, rings, noise"
                )
            })?;

        let result =
            inference::run_inference(&state.model.context_encoder, &input, &state.vit_config);

        serde_json::to_string(&result).map_err(|e| format!("serialization error: {e}"))
    })?
}

/// Run inference on user-provided pixel data.
///
/// Accepts raw pixel data as a flat `Vec<f32>` in CHW order, along with
/// the number of channels, height, and width.
///
/// # Errors
///
/// Returns an error if dimensions don't match or no session exists.
#[wasm_bindgen]
pub fn run_inference_on_data(
    pixel_data: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> Result<String, String> {
    let expected_len = channels * height * width;
    if pixel_data.len() != expected_len {
        return Err(format!(
            "pixel data length {} does not match {}x{}x{} = {}",
            pixel_data.len(),
            channels,
            height,
            width,
            expected_len,
        ));
    }

    let pixel_vec = pixel_data.to_vec();

    with_state(|state| {
        let input = burn::tensor::Tensor::<CpuBackend, 4>::from_floats(
            burn::tensor::TensorData::new(pixel_vec, [1, channels, height, width]),
            &CPU_DEVICE,
        );

        let result =
            inference::run_inference(&state.model.context_encoder, &input, &state.vit_config);

        serde_json::to_string(&result).map_err(|e| format!("serialization error: {e}"))
    })?
}

/// Get information about the current model architecture.
///
/// # Errors
///
/// Returns an error if no training session exists.
#[wasm_bindgen]
pub fn get_model_info() -> Result<String, String> {
    with_state(|state| {
        let info = model_info_from_state(state);
        serde_json::to_string(&info).map_err(|e| format!("serialization error: {e}"))
    })?
}

/// Get the current training step number.
///
/// Returns 0 if no training session exists.
#[wasm_bindgen]
pub fn get_current_step() -> usize {
    TRAINING_STATE.with(|cell| cell.borrow().as_ref().map(|s| s.current_step).unwrap_or(0))
}

/// Reset the training session to step 0 with a fresh model.
///
/// Preserves the existing configuration.
///
/// # Errors
///
/// Returns an error if no training session exists.
#[wasm_bindgen]
pub fn reset_training() -> Result<String, String> {
    let config = with_state(|state| state.config.clone())?;
    let config_json =
        serde_json::to_string(&config).map_err(|e| format!("serialization error: {e}"))?;
    create_training_session(&config_json)
}

// --- Internal helpers ---

#[derive(serde::Serialize)]
struct ModelInfo {
    preset: String,
    embed_dim: usize,
    num_layers: usize,
    num_heads: usize,
    patch_size: (usize, usize),
    image_size: (usize, usize),
    num_patches: usize,
    in_channels: usize,
    total_steps: usize,
    current_step: usize,
}

fn model_info_from_state(state: &TrainingState) -> ModelInfo {
    let vit = &state.vit_config;
    let (ph, pw) = vit.patch_size;
    ModelInfo {
        preset: "tiny_test".to_string(),
        embed_dim: vit.embed_dim,
        num_layers: vit.num_layers,
        num_heads: vit.num_heads,
        patch_size: vit.patch_size,
        image_size: (vit.image_height, vit.image_width),
        num_patches: (vit.image_height / ph) * (vit.image_width / pw),
        in_channels: vit.in_channels,
        total_steps: state.config.total_steps,
        current_step: state.current_step,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::StepMetrics;

    #[test]
    fn test_create_training_session_with_default_config() {
        let config = TrainingConfig::default();
        let config_json = serde_json::to_string(&config).unwrap();
        let result = create_training_session(&config_json);
        assert!(result.is_ok(), "session creation failed: {:?}", result);
    }

    #[test]
    fn test_training_step_returns_metrics() {
        let config = TrainingConfig {
            total_steps: 5,
            warmup_steps: 1,
            batch_size: 1,
            ..TrainingConfig::default()
        };
        let config_json = serde_json::to_string(&config).unwrap();
        create_training_session(&config_json).unwrap();

        let result = training_step();
        assert!(result.is_ok(), "training step failed: {:?}", result);

        let metrics: StepMetrics = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(metrics.step, 0);
        assert!(metrics.total_loss.is_finite());
    }

    #[test]
    fn test_inference_on_pattern() {
        let config = TrainingConfig::default();
        let config_json = serde_json::to_string(&config).unwrap();
        create_training_session(&config_json).unwrap();

        for pattern in &["gradient", "checkerboard", "rings", "noise"] {
            let result = run_inference_on_pattern(pattern);
            assert!(
                result.is_ok(),
                "inference on '{pattern}' failed: {:?}",
                result
            );
        }
    }

    #[test]
    fn test_inference_on_unknown_pattern_errors() {
        let config = TrainingConfig::default();
        let config_json = serde_json::to_string(&config).unwrap();
        create_training_session(&config_json).unwrap();

        let result = run_inference_on_pattern("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_model_info() {
        let config = TrainingConfig::default();
        let config_json = serde_json::to_string(&config).unwrap();
        create_training_session(&config_json).unwrap();

        let result = get_model_info();
        assert!(result.is_ok());

        let info: serde_json::Value = serde_json::from_str(&result.unwrap()).unwrap();
        assert_eq!(info["preset"], "tiny_test");
        assert_eq!(info["embed_dim"], 32);
    }

    #[test]
    fn test_multiple_training_steps_produce_finite_loss() {
        let config = TrainingConfig {
            total_steps: 10,
            warmup_steps: 2,
            batch_size: 1,
            learning_rate: 1e-3,
            ..TrainingConfig::default()
        };
        let config_json = serde_json::to_string(&config).unwrap();
        create_training_session(&config_json).unwrap();

        let mut losses = Vec::new();
        for _ in 0..10 {
            let result = training_step().unwrap();
            let metrics: StepMetrics = serde_json::from_str(&result).unwrap();
            losses.push(metrics.total_loss);
        }

        for (i, loss) in losses.iter().enumerate() {
            assert!(loss.is_finite(), "loss at step {i} is not finite: {loss}");
        }
    }

    #[test]
    fn test_reset_training() {
        let config = TrainingConfig {
            total_steps: 5,
            warmup_steps: 1,
            batch_size: 1,
            ..TrainingConfig::default()
        };
        let config_json = serde_json::to_string(&config).unwrap();
        create_training_session(&config_json).unwrap();

        training_step().unwrap();
        training_step().unwrap();
        assert_eq!(get_current_step(), 2);

        reset_training().unwrap();
        assert_eq!(get_current_step(), 0);
    }

    #[test]
    fn test_run_inference_on_data() {
        let config = TrainingConfig::default();
        let config_json = serde_json::to_string(&config).unwrap();
        create_training_session(&config_json).unwrap();

        let pixels = vec![0.5f32; 64]; // 1 channel * 8 * 8
        let result = run_inference_on_data(&pixels, 1, 8, 8);
        assert!(result.is_ok(), "inference on data failed: {:?}", result);
    }

    #[test]
    fn test_run_inference_on_data_wrong_size_errors() {
        let config = TrainingConfig::default();
        let config_json = serde_json::to_string(&config).unwrap();
        create_training_session(&config_json).unwrap();

        let pixels = vec![0.5f32; 10];
        let result = run_inference_on_data(&pixels, 1, 8, 8);
        assert!(result.is_err());
    }
}
