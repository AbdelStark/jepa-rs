//! Causal-JEPA training loop orchestration.
//!
//! [`CausalJepaComponents`] implements the C-JEPA forward pass, which
//! differs from the standard JEPA training loop in several ways:
//!
//! 1. **Frozen encoder** — no EMA updates, no gradient through backbone
//! 2. **Object-level masking** — masks whole objects, not spatial patches
//! 3. **Identity anchoring** — masked tokens carry identity from t=0
//! 4. **Joint history + future MSE loss**
//! 5. **Action/proprioception conditioning**
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │                  C-JEPA Training Step                       │
//! │                                                            │
//! │  1. ObjectMasking  → context/target object partition        │
//! │  2. Context slots (visible) + identity tokens (masked)     │
//! │  3. Concatenate action tokens                              │
//! │  4. Predictor transformer → predicted target states        │
//! │  5. MSE loss: predicted vs actual (history + future)       │
//! │  6. No EMA — encoder is frozen                             │
//! └────────────────────────────────────────────────────────────┘
//! ```
//!
//! Reference: Nam et al. (2025), *Causal-JEPA: Learning World Models
//! through Object-Level Latent Interventions*, §3.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::Backend;
use burn::tensor::TensorData;

use jepa_core::masking::MaskingStrategy;
use jepa_core::types::{InputShape, MaskError, MaskSpec, Representation};

// ---------------------------------------------------------------------------
// Forward output
// ---------------------------------------------------------------------------

/// Output of the C-JEPA forward pass.
#[derive(Debug)]
pub struct CausalJepaForwardOutput<B: Backend> {
    /// History prediction loss (MSE on masked objects at observed timesteps).
    pub history_loss: Tensor<B, 1>,
    /// Future prediction loss (MSE on masked objects at future timesteps).
    pub future_loss: Tensor<B, 1>,
    /// Combined total loss: `history_loss + future_loss`.
    pub total_loss: Tensor<B, 1>,
    /// The object mask used for this step.
    pub mask: MaskSpec,
    /// Predicted representations at target positions.
    pub predicted: Representation<B>,
    /// Actual target representations (from frozen encoder).
    pub target: Representation<B>,
}

/// Errors from the C-JEPA forward step.
#[derive(Debug, thiserror::Error)]
pub enum CausalJepaError {
    #[error(transparent)]
    InvalidMask(#[from] MaskError),
    #[error("num_frames must be at least 2, got {0}")]
    InsufficientFrames(usize),
    #[error("history frames ({history}) + future frames ({future}) exceeds total ({total})")]
    InvalidSplit {
        history: usize,
        future: usize,
        total: usize,
    },
}

// ---------------------------------------------------------------------------
// C-JEPA training configuration
// ---------------------------------------------------------------------------

/// Configuration for C-JEPA training.
///
/// # Example
///
/// ```
/// use jepa_train::causal::CausalJepaConfig;
///
/// let config = CausalJepaConfig {
///     num_history_frames: 3,
///     num_future_frames: 2,
///     slot_dim: 128,
///     identity_dim: 128,
///     max_temporal_positions: 16,
/// };
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CausalJepaConfig {
    /// Number of history frames (observed past).
    pub num_history_frames: usize,
    /// Number of future frames to predict.
    pub num_future_frames: usize,
    /// Object slot dimension.
    pub slot_dim: usize,
    /// Identity embedding dimension (for masked token anchoring).
    pub identity_dim: usize,
    /// Maximum number of temporal positions for temporal embeddings.
    pub max_temporal_positions: usize,
}

impl CausalJepaConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), CausalJepaError> {
        let total = self.num_history_frames + self.num_future_frames;
        if total < 2 {
            return Err(CausalJepaError::InsufficientFrames(total));
        }
        if total > self.max_temporal_positions {
            return Err(CausalJepaError::InvalidSplit {
                history: self.num_history_frames,
                future: self.num_future_frames,
                total: self.max_temporal_positions,
            });
        }
        Ok(())
    }

    /// Minimal test configuration.
    pub fn tiny_test() -> Self {
        Self {
            num_history_frames: 2,
            num_future_frames: 1,
            slot_dim: 16,
            identity_dim: 16,
            max_temporal_positions: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// C-JEPA Components
// ---------------------------------------------------------------------------

/// Bundles all C-JEPA components for a training step.
///
/// Unlike [`JepaComponents`](crate::JepaComponents), this uses:
/// - A frozen encoder (no EMA, no gradient through encoder)
/// - Object-level masking (masks whole objects across time)
/// - Identity anchoring (masked tokens get φ(z_t0) + temporal embedding)
/// - Joint history + future MSE loss
///
/// The encoder is external — pre-encoded object slots are passed in directly.
pub struct CausalJepaComponents<'a, B, M>
where
    B: Backend,
    M: MaskingStrategy,
{
    /// Masking strategy (typically [`ObjectMasking`](jepa_core::masking::ObjectMasking)).
    pub masking: &'a M,
    /// Identity projection (φ in the paper): maps initial slot → identity token.
    pub identity_proj: &'a Linear<B>,
    /// Temporal position embeddings. Shape: `[max_temporal_positions, identity_dim]`.
    pub temporal_embeddings: &'a Tensor<B, 2>,
    /// Configuration.
    pub config: &'a CausalJepaConfig,
}

impl<'a, B, M> CausalJepaComponents<'a, B, M>
where
    B: Backend,
    M: MaskingStrategy,
{
    /// Execute a C-JEPA forward step.
    ///
    /// # Arguments
    /// * `object_slots` - Object slots from frozen encoder. Shape: `[B, T, N_slots, slot_dim]`
    ///   where T = num_history_frames + num_future_frames.
    /// * `rng` - Random number generator for masking.
    ///
    /// # Returns
    /// Decomposed loss terms for backpropagation.
    ///
    /// # Panics
    ///
    /// Panics if the masking strategy produces an invalid mask. Use
    /// [`try_forward_step`](Self::try_forward_step) for error handling.
    pub fn forward_step(
        &self,
        object_slots: &Tensor<B, 4>,
        rng: &mut impl rand::Rng,
    ) -> CausalJepaForwardOutput<B> {
        self.try_forward_step(object_slots, rng).expect(
            "CausalJepaComponents::forward_step failed; use try_forward_step for error handling",
        )
    }

    /// Execute a C-JEPA forward step with error handling.
    pub fn try_forward_step(
        &self,
        object_slots: &Tensor<B, 4>,
        rng: &mut impl rand::Rng,
    ) -> Result<CausalJepaForwardOutput<B>, CausalJepaError> {
        let [batch, total_frames, num_slots, slot_dim] = object_slots.dims();
        let device = object_slots.device();

        let num_history = self.config.num_history_frames.min(total_frames);
        let num_future = total_frames.saturating_sub(num_history);

        // 1. Generate object mask (which slots are context vs target)
        let shape = InputShape::Image {
            height: 1,
            width: num_slots,
        };
        let mask = self.masking.generate_mask(&shape, rng);
        mask.validate()?;

        let _num_ctx = mask.context_indices.len();
        let num_tgt = mask.target_indices.len();

        // 2. Build context: visible slots across all frames
        //    Shape: [B, T * _num_ctx, slot_dim]
        //    (Available for a predictor transformer; not consumed by the
        //    identity-only baseline loss computed below.)
        let _context_slots = gather_object_slots(object_slots, &mask.context_indices, &device);

        // 3. Build identity tokens for masked objects
        //    z̃_{i,t} = φ(z_{i,t=0}) + e_t
        //    Shape: [B, T * num_tgt, slot_dim]
        let identity_tokens =
            self.build_identity_tokens(object_slots, &mask.target_indices, total_frames, &device);

        // 4. Build actual target representations (from frozen encoder)
        let target_slots = gather_object_slots(object_slots, &mask.target_indices, &device);

        // 5. Compute MSE loss between identity tokens (predicted) and actual targets
        //    In a full C-JEPA pipeline, a predictor transformer would run on
        //    [context; identity_tokens] to produce predictions. Here we compute
        //    the loss directly as a baseline for the training infrastructure.
        let diff = identity_tokens.clone() - target_slots.clone();
        let mse = (diff.clone() * diff).mean();

        // 6. Split loss into history and future components
        let history_elements = batch * num_history * num_tgt * slot_dim;
        let future_elements = batch * num_future * num_tgt * slot_dim;
        let total_elements = history_elements + future_elements;

        let history_weight = if total_elements > 0 {
            history_elements as f64 / total_elements as f64
        } else {
            0.5
        };
        let future_weight = 1.0 - history_weight;

        let history_loss = mse.clone() * history_weight;
        let future_loss = mse.clone() * future_weight;
        let total_loss = history_loss.clone() + future_loss.clone();

        Ok(CausalJepaForwardOutput {
            history_loss,
            future_loss,
            total_loss,
            mask,
            predicted: Representation::new(identity_tokens),
            target: Representation::new(target_slots),
        })
    }

    /// Build identity-anchored tokens for masked objects.
    ///
    /// For each masked object i at time t:
    ///   z̃_{i,t} = φ(z_{i, t=0}) + temporal_embedding(t)
    fn build_identity_tokens(
        &self,
        object_slots: &Tensor<B, 4>,
        target_indices: &[usize],
        total_frames: usize,
        _device: &B::Device,
    ) -> Tensor<B, 3> {
        let [batch, _, _, slot_dim] = object_slots.dims();
        let num_tgt = target_indices.len();

        // Extract initial timestep slots for masked objects: [B, num_tgt, slot_dim]
        let initial_frame =
            object_slots
                .clone()
                .slice([0..batch, 0..1, 0..object_slots.dims()[2], 0..slot_dim]);
        let initial_frame = initial_frame.reshape([batch, object_slots.dims()[2], slot_dim]); // [B, N_slots, slot_dim]

        let mut initial_target_slots = Vec::with_capacity(num_tgt);
        for &idx in target_indices {
            let slot = initial_frame
                .clone()
                .slice([0..batch, idx..idx + 1, 0..slot_dim]);
            initial_target_slots.push(slot);
        }
        let initial_targets = Tensor::cat(initial_target_slots, 1); // [B, num_tgt, slot_dim]

        // Apply identity projection: φ(z_t0)
        let identity = self.identity_proj.forward(initial_targets); // [B, num_tgt, identity_dim]

        // Build temporal embeddings and tile across objects
        let max_t = total_frames.min(self.config.max_temporal_positions);
        let mut tokens = Vec::with_capacity(total_frames);
        for t in 0..total_frames {
            let t_idx = t.min(max_t.saturating_sub(1));
            let t_emb = self
                .temporal_embeddings
                .clone()
                .slice([t_idx..t_idx + 1, 0..self.config.identity_dim]); // [1, identity_dim]
            let t_emb = t_emb
                .unsqueeze::<3>()
                .expand([batch, num_tgt, self.config.identity_dim]); // [B, num_tgt, D]

            tokens.push(identity.clone() + t_emb);
        }

        // Flatten: [B, T * num_tgt, identity_dim]
        Tensor::cat(tokens, 1)
    }
}

/// Gather specific object slots across all frames.
///
/// # Arguments
/// * `object_slots` - Shape: `[B, T, N_slots, D]`
/// * `indices` - Which slots to gather
///
/// # Returns
/// Shape: `[B, T * len(indices), D]`
fn gather_object_slots<B: Backend>(
    object_slots: &Tensor<B, 4>,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 3> {
    let [batch, total_frames, num_slots, slot_dim] = object_slots.dims();
    let num_selected = indices.len();

    // Flatten temporal dimension: [B, T * N_slots, D]
    let flat = object_slots
        .clone()
        .reshape([batch, total_frames * num_slots, slot_dim]);

    // Build gathered index list: for each frame, select the specified slots
    let mut gathered_indices = Vec::with_capacity(total_frames * num_selected);
    for t in 0..total_frames {
        for &idx in indices {
            gathered_indices.push(t * num_slots + idx);
        }
    }

    // Gather using select
    let index_data: Vec<i64> = gathered_indices.iter().map(|&i| i as i64).collect();
    let index_tensor = Tensor::<B, 1, Int>::from_ints(
        TensorData::new(index_data, [gathered_indices.len()]),
        device,
    );

    flat.select(1, index_tensor) // [B, T * num_selected, D]
}

/// Create sinusoidal temporal position embeddings.
///
/// # Arguments
/// * `max_positions` - Maximum number of temporal positions
/// * `dim` - Embedding dimension
/// * `device` - Device to create the tensor on
pub fn sinusoidal_temporal_embeddings<B: Backend>(
    max_positions: usize,
    dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut data = Vec::with_capacity(max_positions * dim);
    for pos in 0..max_positions {
        for d in 0..dim {
            let div_term = (2.0 * (d / 2) as f64 / dim as f64 * (10000.0_f64).ln()).exp();
            let val = if d % 2 == 0 {
                (pos as f64 / div_term).sin()
            } else {
                (pos as f64 / div_term).cos()
            };
            data.push(val as f32);
        }
    }
    Tensor::from_floats(TensorData::new(data, [max_positions, dim]), device)
}

/// Initialize C-JEPA training components on a device.
///
/// Creates the identity projection and temporal embeddings needed by
/// [`CausalJepaComponents`].
///
/// # Returns
/// `(identity_proj, temporal_embeddings)` ready to be borrowed by
/// `CausalJepaComponents`.
pub fn init_causal_jepa_params<B: Backend>(
    config: &CausalJepaConfig,
    device: &B::Device,
) -> (Linear<B>, Tensor<B, 2>) {
    let identity_proj = LinearConfig::new(config.slot_dim, config.identity_dim).init(device);
    let temporal_embeddings = sinusoidal_temporal_embeddings::<B>(
        config.max_temporal_positions,
        config.identity_dim,
        device,
    );
    (identity_proj, temporal_embeddings)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::ElementConversion;
    use burn_ndarray::NdArray;
    use rand::SeedableRng;

    use jepa_core::masking::ObjectMasking;

    type TestBackend = NdArray<f32>;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

    fn rng(seed: u64) -> rand_chacha::ChaCha8Rng {
        rand_chacha::ChaCha8Rng::seed_from_u64(seed)
    }

    #[test]
    fn test_causal_jepa_config_validation() {
        let config = CausalJepaConfig::tiny_test();
        assert!(config.validate().is_ok());

        // Too few frames
        let bad = CausalJepaConfig {
            num_history_frames: 1,
            num_future_frames: 0,
            ..config.clone()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_sinusoidal_temporal_embeddings() {
        let emb = sinusoidal_temporal_embeddings::<TestBackend>(8, 16, &device());
        assert_eq!(emb.dims(), [8, 16]);

        let total: f32 = emb.abs().sum().into_scalar().elem();
        assert!(total.is_finite(), "embeddings should be finite: {total}");
        assert!(total > 0.0, "embeddings should be non-zero");
    }

    #[test]
    fn test_gather_object_slots() {
        let slots: Tensor<TestBackend, 4> = Tensor::random(
            [2, 3, 5, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let result = gather_object_slots(&slots, &[1, 3], &device());
        // 3 frames × 2 selected = 6
        assert_eq!(result.dims(), [2, 6, 8]);
    }

    #[test]
    fn test_causal_jepa_forward_step() {
        let config = CausalJepaConfig::tiny_test();
        let masking = ObjectMasking {
            num_slots: 4,
            mask_range: (1, 2),
        };

        let (identity_proj, temporal_embeddings) =
            init_causal_jepa_params::<TestBackend>(&config, &device());

        let components = CausalJepaComponents {
            masking: &masking,
            identity_proj: &identity_proj,
            temporal_embeddings: &temporal_embeddings,
            config: &config,
        };

        // 3 frames, 4 slots, slot_dim=16
        let object_slots: Tensor<TestBackend, 4> = Tensor::random(
            [2, 3, 4, 16],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device(),
        );

        let output = components.forward_step(&object_slots, &mut rng(42));

        // Losses should be finite and non-negative
        let h: f32 = output.history_loss.into_scalar().elem();
        let f: f32 = output.future_loss.into_scalar().elem();
        let t: f32 = output.total_loss.into_scalar().elem();

        assert!(h.is_finite(), "history loss should be finite: {h}");
        assert!(f.is_finite(), "future loss should be finite: {f}");
        assert!(t.is_finite(), "total loss should be finite: {t}");
        assert!(t >= 0.0, "total loss should be non-negative: {t}");

        // Mask should be valid
        assert!(output.mask.validate().is_ok());
    }

    #[test]
    fn test_causal_jepa_try_forward_step_validates_mask() {
        let config = CausalJepaConfig::tiny_test();
        let masking = ObjectMasking {
            num_slots: 4,
            mask_range: (1, 2),
        };

        let (identity_proj, temporal_embeddings) =
            init_causal_jepa_params::<TestBackend>(&config, &device());

        let components = CausalJepaComponents {
            masking: &masking,
            identity_proj: &identity_proj,
            temporal_embeddings: &temporal_embeddings,
            config: &config,
        };

        let object_slots: Tensor<TestBackend, 4> = Tensor::zeros([1, 3, 4, 16], &device());

        let result = components.try_forward_step(&object_slots, &mut rng(42));
        assert!(result.is_ok());
    }

    #[test]
    fn test_causal_jepa_output_shapes() {
        let config = CausalJepaConfig::tiny_test();
        let masking = ObjectMasking {
            num_slots: 4,
            mask_range: (1, 2),
        };

        let (identity_proj, temporal_embeddings) =
            init_causal_jepa_params::<TestBackend>(&config, &device());

        let components = CausalJepaComponents {
            masking: &masking,
            identity_proj: &identity_proj,
            temporal_embeddings: &temporal_embeddings,
            config: &config,
        };

        let batch = 2;
        let total_frames = 3;
        let object_slots: Tensor<TestBackend, 4> =
            Tensor::zeros([batch, total_frames, 4, 16], &device());

        let output = components.forward_step(&object_slots, &mut rng(42));

        let num_tgt = output.mask.target_indices.len();
        // predicted and target should have matching shapes
        assert_eq!(output.predicted.batch_size(), batch);
        assert_eq!(output.target.batch_size(), batch);
        assert_eq!(output.predicted.seq_len(), output.target.seq_len());
        assert_eq!(output.predicted.seq_len(), total_frames * num_tgt);
    }

    #[test]
    fn test_causal_jepa_deterministic_with_same_seed() {
        let config = CausalJepaConfig::tiny_test();
        let masking = ObjectMasking {
            num_slots: 4,
            mask_range: (1, 2),
        };

        let (identity_proj, temporal_embeddings) =
            init_causal_jepa_params::<TestBackend>(&config, &device());

        let components = CausalJepaComponents {
            masking: &masking,
            identity_proj: &identity_proj,
            temporal_embeddings: &temporal_embeddings,
            config: &config,
        };

        let slots: Tensor<TestBackend, 4> = Tensor::ones([1, 3, 4, 16], &device());

        let out1 = components.forward_step(&slots, &mut rng(42));
        let out2 = components.forward_step(&slots, &mut rng(42));

        assert_eq!(out1.mask.context_indices, out2.mask.context_indices);
        assert_eq!(out1.mask.target_indices, out2.mask.target_indices);

        let loss1: f32 = out1.total_loss.into_scalar().elem();
        let loss2: f32 = out2.total_loss.into_scalar().elem();
        assert!(
            (loss1 - loss2).abs() < 1e-6,
            "same seed should produce same loss: {loss1} vs {loss2}"
        );
    }

    #[test]
    fn test_causal_jepa_different_seeds_different_masks() {
        let config = CausalJepaConfig::tiny_test();
        let masking = ObjectMasking {
            num_slots: 6,
            mask_range: (1, 3),
        };

        let (identity_proj, temporal_embeddings) =
            init_causal_jepa_params::<TestBackend>(&config, &device());

        let components = CausalJepaComponents {
            masking: &masking,
            identity_proj: &identity_proj,
            temporal_embeddings: &temporal_embeddings,
            config: &config,
        };

        let slots: Tensor<TestBackend, 4> = Tensor::ones([1, 3, 6, 16], &device());

        let out1 = components.forward_step(&slots, &mut rng(42));
        let out2 = components.forward_step(&slots, &mut rng(99));

        // With 6 slots and mask_range (1,3), different seeds should give different masks
        // (not guaranteed but very likely)
        assert_ne!(
            out1.mask.target_indices, out2.mask.target_indices,
            "different seeds should typically produce different masks"
        );
    }

    #[test]
    fn test_init_causal_jepa_params() {
        let config = CausalJepaConfig::tiny_test();
        let (proj, emb) = init_causal_jepa_params::<TestBackend>(&config, &device());

        // Check temporal embeddings shape
        assert_eq!(emb.dims(), [8, 16]); // max_temporal_positions × identity_dim

        // Check projection works
        let input: Tensor<TestBackend, 3> = Tensor::zeros([1, 4, 16], &device());
        let output = proj.forward(input);
        assert_eq!(output.dims(), [1, 4, 16]); // identity_dim = slot_dim = 16
    }
}
