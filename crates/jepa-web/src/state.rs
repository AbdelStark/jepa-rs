//! Shared model and training state for the browser demo.
//!
//! Holds the I-JEPA model, optimizer, schedule, and training metrics in a
//! structure that can be created once and mutated across JS-driven training
//! steps.

use burn::optim::AdamWConfig;
use serde::{Deserialize, Serialize};

use jepa_core::ema::Ema;
use jepa_core::masking::BlockMasking;
use jepa_core::types::InputShape;
use jepa_train::schedule::WarmupCosineSchedule;
use jepa_vision::image::{IJepa, IJepaConfig, TransformerPredictorConfig};
use jepa_vision::vit::VitConfig;

use crate::backend::CpuBackend;

/// The concrete optimizer type used in the web demo.
pub type WebOptimizer =
    burn::optim::adaptor::OptimizerAdaptor<burn::optim::AdamW, IJepa<CpuBackend>, CpuBackend>;

/// Configuration sent from JavaScript to initialize a training session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Peak learning rate.
    pub learning_rate: f64,
    /// Batch size per training step.
    pub batch_size: usize,
    /// Total training steps.
    pub total_steps: usize,
    /// Number of warmup steps.
    pub warmup_steps: usize,
    /// Base EMA momentum.
    pub ema_momentum: f64,
    /// Regularization weight.
    pub reg_weight: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            batch_size: 2,
            total_steps: 200,
            warmup_steps: 20,
            ema_momentum: 0.996,
            reg_weight: 1.0,
        }
    }
}

/// Metrics returned to JavaScript after each training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    /// Current step number.
    pub step: usize,
    /// Prediction energy loss.
    pub energy: f64,
    /// Regularization loss.
    pub regularization: f64,
    /// Total loss (energy + reg_weight * regularization).
    pub total_loss: f64,
    /// Learning rate used at this step.
    pub learning_rate: f64,
    /// EMA momentum at this step.
    pub ema_momentum: f64,
}

/// Persistent training state held between JS-driven steps.
pub struct TrainingState {
    /// The I-JEPA model (context encoder, target encoder, predictor).
    pub model: IJepa<CpuBackend>,
    /// AdamW optimizer.
    pub optimizer: WebOptimizer,
    /// Learning rate schedule.
    pub lr_schedule: WarmupCosineSchedule,
    /// EMA updater for target encoder.
    pub ema: Ema,
    /// Block masking strategy.
    pub masking: BlockMasking,
    /// Patch grid shape for mask generation.
    pub input_shape: InputShape,
    /// ViT config (needed for synthetic data generation).
    pub vit_config: VitConfig,
    /// Current training step.
    pub current_step: usize,
    /// Training config.
    pub config: TrainingConfig,
    /// Deterministic RNG.
    pub rng: rand_chacha::ChaCha8Rng,
}

/// Create a `VitConfig` for the tiny test preset (used for browser training).
pub fn tiny_vit_config() -> VitConfig {
    VitConfig::tiny_test()
}

/// Build an I-JEPA model and all training state from a config.
pub fn init_training_state(
    config: TrainingConfig,
    device: &burn_ndarray::NdArrayDevice,
) -> TrainingState {
    use burn::module::Module;
    use rand::SeedableRng;

    let vit_config = tiny_vit_config();
    let embed_dim = vit_config.embed_dim;
    let (patch_h, patch_w) = vit_config.patch_size;
    let grid_h = vit_config.image_height / patch_h;
    let grid_w = vit_config.image_width / patch_w;

    let predictor_config = TransformerPredictorConfig {
        encoder_embed_dim: embed_dim,
        predictor_embed_dim: embed_dim / 2,
        num_layers: 1,
        num_heads: 2,
        max_target_len: grid_h * grid_w,
    };

    let ijepa_config = IJepaConfig {
        encoder: vit_config.clone(),
        predictor: predictor_config,
    };

    let mut model: IJepa<CpuBackend> = ijepa_config.init(device);
    model.target_encoder = model.target_encoder.no_grad();

    let optimizer = AdamWConfig::new().init::<CpuBackend, IJepa<CpuBackend>>();
    let lr_schedule = WarmupCosineSchedule::new(
        config.learning_rate,
        config.warmup_steps,
        config.total_steps,
    );
    let ema = Ema::with_cosine_schedule(config.ema_momentum, config.total_steps);

    let masking = BlockMasking {
        num_targets: 4,
        target_scale: (0.15, 0.2),
        target_aspect_ratio: (0.75, 1.5),
    };

    let input_shape = InputShape::Image {
        height: grid_h,
        width: grid_w,
    };

    let rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

    TrainingState {
        model,
        optimizer,
        lr_schedule,
        ema,
        masking,
        input_shape,
        vit_config,
        current_step: 0,
        config,
        rng,
    }
}
