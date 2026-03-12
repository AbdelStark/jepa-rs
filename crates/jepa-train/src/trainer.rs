//! JEPA training step orchestration.
//!
//! Implements RFC-008 (Training Loop) — the core training step that ties
//! together all JEPA components: masking, encoding, prediction, energy
//! computation, collapse prevention, and EMA update.
//!
//! The trainer computes the forward pass and returns the loss for backprop.
//! It does NOT own the optimizer — that is the caller's responsibility,
//! following burn's convention of keeping optimization separate from
//! model logic.
//!
//! Important: [`JepaComponents::forward_step`] is intentionally generic over
//! `Encoder`, which means it cannot remove hidden targets before encoder
//! self-attention runs. For strict pre-encoder masking semantics, use the
//! modality-specific helpers in `jepa-vision`:
//! `jepa_vision::image::IJepa::forward_step_strict` and
//! `jepa_vision::video::VJepa::forward_step_strict`.

use burn::tensor::{backend::Backend, Tensor, TensorData};

use jepa_core::collapse::CollapseRegularizer;
use jepa_core::ema::Ema;
use jepa_core::energy::EnergyFn;
use jepa_core::masking::MaskingStrategy;
use jepa_core::types::{Energy, InputShape, MaskSpec, Representation};
use jepa_core::{Encoder, Predictor};

use crate::schedule::LrSchedule;

/// Output of the JEPA forward pass (before backprop).
///
/// Contains the decomposed loss terms for the caller to use in
/// gradient computation and logging.
#[derive(Debug)]
pub struct JepaForwardOutput<B: Backend> {
    /// Prediction energy (main loss signal). Shape: `[1]`
    pub energy: Energy<B>,
    /// Collapse prevention regularization loss. Shape: `[1]`
    pub regularization: Tensor<B, 1>,
    /// Total loss (energy + weighted regularization). Shape: `[1]`
    pub total_loss: Tensor<B, 1>,
    /// The mask used for this step (for logging/debugging).
    pub mask: MaskSpec,
    /// Predicted target representations.
    pub predicted: Representation<B>,
    /// Actual target representations (from target encoder, detached).
    pub target: Representation<B>,
}

/// Bundles all JEPA components needed for a training step.
///
/// This struct borrows the model components and configuration needed
/// to execute a single JEPA forward pass. Using a struct avoids
/// excessive function parameters.
pub struct JepaComponents<'a, B, E, P, EF, CR, M>
where
    B: Backend,
    E: Encoder<B>,
    P: Predictor<B>,
    EF: EnergyFn<B>,
    CR: CollapseRegularizer<B>,
    M: MaskingStrategy,
{
    /// The context encoder (trained via gradient descent).
    pub context_encoder: &'a E,
    /// The target encoder (updated via EMA, not gradients).
    pub target_encoder: &'a E,
    /// The predictor network.
    pub predictor: &'a P,
    /// Energy function for measuring prediction quality.
    pub energy_fn: &'a EF,
    /// Collapse prevention regularizer.
    pub regularizer: &'a CR,
    /// Masking strategy for generating context/target splits.
    pub masking: &'a M,
    /// Weight for the regularization loss term.
    pub reg_weight: f64,
    /// Marker for backend type.
    _backend: std::marker::PhantomData<B>,
}

impl<'a, B, E, P, EF, CR, M> JepaComponents<'a, B, E, P, EF, CR, M>
where
    B: Backend,
    E: Encoder<B>,
    P: Predictor<B>,
    EF: EnergyFn<B>,
    CR: CollapseRegularizer<B>,
    M: MaskingStrategy,
{
    /// Create a new set of JEPA training components.
    pub fn new(
        context_encoder: &'a E,
        target_encoder: &'a E,
        predictor: &'a P,
        energy_fn: &'a EF,
        regularizer: &'a CR,
        masking: &'a M,
        reg_weight: f64,
    ) -> Self {
        Self {
            context_encoder,
            target_encoder,
            predictor,
            energy_fn,
            regularizer,
            masking,
            reg_weight,
            _backend: std::marker::PhantomData,
        }
    }

    /// Execute a single JEPA training step forward pass.
    ///
    /// Orchestrates the full JEPA training pipeline per RFC-008:
    /// 1. Generate a mask to split input into context/target
    /// 2. Encode context with the context encoder (gradients flow)
    /// 3. Encode targets with the target encoder (no gradient update — EMA only)
    /// 4. Predict target representations from context using the predictor
    /// 5. Compute energy (prediction loss) between predicted and actual targets
    /// 6. Compute collapse prevention regularization loss
    /// 7. Combine into total loss
    ///
    /// This helper is an approximation of strict JEPA masking semantics:
    /// the context encoder still sees the full token sequence before the
    /// context slice is gathered. Prefer the encoder-specific strict helpers
    /// when hidden-token isolation matters.
    ///
    /// The caller is responsible for:
    /// - Running backward pass on `total_loss`
    /// - Stepping the optimizer
    /// - Performing the EMA update of the target encoder
    pub fn forward_step(
        &self,
        input: &E::Input,
        input_shape: &InputShape,
        rng: &mut impl rand::Rng,
    ) -> JepaForwardOutput<B> {
        // 1. Generate mask
        let mask = self.masking.generate_mask(input_shape, rng);
        mask.validate()
            .expect("masking strategy must produce a valid mask");

        // The generic encoder trait doesn't let the trainer remove masked tokens
        // before encoder attention runs, so this orchestration layer filters
        // tokens immediately after encoding. Encoder-specific training code
        // should prefer pre-encoder masking when strict JEPA semantics matter.
        let context_repr = self.context_encoder.encode(input);
        let context_slice = context_repr.gather(&mask.context_indices);

        // 3. Encode full input with target encoder (no gradient update)
        //    Gradient detachment is the caller's responsibility via burn's autodiff.
        let target_repr = self.target_encoder.encode(input);
        let target_slice = target_repr.gather(&mask.target_indices);

        // 4. Predict target representations from context
        let batch = context_slice.batch_size();
        let num_targets = mask.target_indices.len();
        let device = context_slice.embeddings.device();
        let target_positions = target_positions_tensor::<B>(&mask.target_indices, batch, &device);

        let predicted = self
            .predictor
            .predict(&context_slice, &target_positions, None);

        // 6. Compute energy (prediction loss)
        let energy = self.energy_fn.compute(&predicted, &target_slice);

        // 7. Compute collapse prevention regularization
        //    Flatten to [batch * seq_len, embed_dim] for VICReg
        let embed_dim = target_slice.embed_dim();
        let pred_flat = predicted
            .embeddings
            .clone()
            .reshape([batch * num_targets, embed_dim]);
        let target_flat = target_slice
            .embeddings
            .clone()
            .reshape([batch * num_targets, embed_dim]);
        let regularization = self.regularizer.loss(&pred_flat, &target_flat);

        // 8. Combine losses
        let total_loss = energy.value.clone() + regularization.clone() * self.reg_weight;

        JepaForwardOutput {
            energy,
            regularization,
            total_loss,
            mask,
            predicted,
            target: target_slice,
        }
    }
}

/// Metadata for a completed training step (after optimizer step + EMA update).
#[derive(Debug, Clone)]
pub struct StepReport {
    /// Training step number.
    pub step: usize,
    /// Energy (prediction loss) value.
    pub energy: f64,
    /// Regularization loss value.
    pub regularization: f64,
    /// Total loss value.
    pub total_loss: f64,
    /// Learning rate used for this step.
    pub learning_rate: f64,
    /// EMA momentum used for this step.
    pub ema_momentum: f64,
    /// Mask ratio (fraction of tokens masked as targets).
    pub mask_ratio: f64,
}

/// Compute the learning rate and EMA momentum for a given step.
///
/// Convenience function for the training loop to query schedule values.
pub fn schedule_values(lr_schedule: &dyn LrSchedule, ema: &Ema, step: usize) -> (f64, f64) {
    (lr_schedule.get_lr(step), ema.get_momentum(step))
}

fn target_positions_tensor<B: Backend>(
    indices: &[usize],
    batch: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let num_targets = indices.len();
    let mut position_data = Vec::with_capacity(batch * num_targets);
    for _ in 0..batch {
        position_data.extend(indices.iter().map(|&index| index as f32));
    }

    Tensor::from_floats(TensorData::new(position_data, [batch, num_targets]), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::ElementConversion;
    use burn_ndarray::NdArray;
    use rand::SeedableRng;

    use jepa_core::collapse::VICReg;
    use jepa_core::energy::L2Energy;
    use jepa_core::masking::BlockMasking;

    type TestBackend = NdArray<f32>;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

    /// Trivial encoder for testing the training step.
    struct TestEncoder {
        embed_dim: usize,
    }

    impl Encoder<TestBackend> for TestEncoder {
        type Input = Tensor<TestBackend, 4>;

        fn encode(&self, input: &Self::Input) -> Representation<TestBackend> {
            let [batch, _c, h, w] = input.dims();
            let seq_len = h * w;
            Representation::new(Tensor::ones(
                [batch, seq_len, self.embed_dim],
                &input.device(),
            ))
        }

        fn embed_dim(&self) -> usize {
            self.embed_dim
        }
    }

    /// Trivial predictor for testing.
    struct TestPredictor {
        embed_dim: usize,
    }

    impl Predictor<TestBackend> for TestPredictor {
        fn predict(
            &self,
            _context: &Representation<TestBackend>,
            target_positions: &Tensor<TestBackend, 2>,
            _latent: Option<&Tensor<TestBackend, 2>>,
        ) -> Representation<TestBackend> {
            let [batch, num_targets] = target_positions.dims();
            Representation::new(Tensor::ones(
                [batch, num_targets, self.embed_dim],
                &target_positions.device(),
            ))
        }
    }

    struct PositionAwarePredictor;

    impl Predictor<TestBackend> for PositionAwarePredictor {
        fn predict(
            &self,
            _context: &Representation<TestBackend>,
            target_positions: &Tensor<TestBackend, 2>,
            _latent: Option<&Tensor<TestBackend, 2>>,
        ) -> Representation<TestBackend> {
            let [batch, num_targets] = target_positions.dims();
            Representation::new(target_positions.clone().reshape([batch, num_targets, 1]))
        }
    }

    struct FixedMasking {
        mask: MaskSpec,
    }

    impl MaskingStrategy for FixedMasking {
        fn generate_mask(&self, _shape: &InputShape, _rng: &mut impl rand::Rng) -> MaskSpec {
            self.mask.clone()
        }
    }

    fn make_components<'a>(
        context_encoder: &'a TestEncoder,
        target_encoder: &'a TestEncoder,
        predictor: &'a TestPredictor,
        energy_fn: &'a L2Energy,
        regularizer: &'a VICReg,
        masking: &'a BlockMasking,
        reg_weight: f64,
    ) -> JepaComponents<'a, TestBackend, TestEncoder, TestPredictor, L2Energy, VICReg, BlockMasking>
    {
        JepaComponents::new(
            context_encoder,
            target_encoder,
            predictor,
            energy_fn,
            regularizer,
            masking,
            reg_weight,
        )
    }

    #[test]
    fn test_jepa_forward_step_runs() {
        let embed_dim = 16;
        let context_encoder = TestEncoder { embed_dim };
        let target_encoder = TestEncoder { embed_dim };
        let predictor = TestPredictor { embed_dim };
        let energy_fn = L2Energy;
        let regularizer = VICReg::default();
        let masking = BlockMasking {
            num_targets: 2,
            target_scale: (0.15, 0.3),
            target_aspect_ratio: (0.75, 1.5),
        };

        let components = make_components(
            &context_encoder,
            &target_encoder,
            &predictor,
            &energy_fn,
            &regularizer,
            &masking,
            1.0,
        );

        let input: Tensor<TestBackend, 4> = Tensor::ones([2, 1, 4, 4], &device());
        let input_shape = InputShape::Image {
            height: 4,
            width: 4,
        };

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let output = components.forward_step(&input, &input_shape, &mut rng);

        // Energy should be finite
        let energy_val: f32 = output.energy.value.into_scalar().elem();
        assert!(
            energy_val.is_finite(),
            "energy should be finite: {energy_val}"
        );

        // Regularization should be finite
        let reg_val: f32 = output.regularization.into_scalar().elem();
        assert!(
            reg_val.is_finite(),
            "regularization should be finite: {reg_val}"
        );

        // Total loss should be finite
        let total_val: f32 = output.total_loss.into_scalar().elem();
        assert!(
            total_val.is_finite(),
            "total loss should be finite: {total_val}"
        );

        // Mask should be valid
        assert!(output.mask.validate().is_ok());
    }

    #[test]
    fn test_jepa_forward_step_passes_real_target_positions() {
        let embed_dim = 1;
        let context_encoder = TestEncoder { embed_dim };
        let target_encoder = TestEncoder { embed_dim };
        let predictor = PositionAwarePredictor;
        let energy_fn = L2Energy;
        let regularizer = VICReg::default();
        let masking = FixedMasking {
            mask: MaskSpec {
                context_indices: vec![0, 2],
                target_indices: vec![1, 3],
                total_tokens: 4,
            },
        };

        let components = JepaComponents::new(
            &context_encoder,
            &target_encoder,
            &predictor,
            &energy_fn,
            &regularizer,
            &masking,
            0.0,
        );

        let input: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 2, 2], &device());
        let input_shape = InputShape::Image {
            height: 2,
            width: 2,
        };

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let output = components.forward_step(&input, &input_shape, &mut rng);
        let values: Vec<f32> = output.predicted.embeddings.into_data().to_vec().unwrap();

        assert_eq!(values.len(), 2);
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_jepa_forward_step_energy_is_non_negative() {
        let embed_dim = 8;
        let context_encoder = TestEncoder { embed_dim };
        let target_encoder = TestEncoder { embed_dim };
        let predictor = TestPredictor { embed_dim };
        let energy_fn = L2Energy;
        let regularizer = VICReg::default();
        let masking = BlockMasking {
            num_targets: 1,
            target_scale: (0.1, 0.3),
            target_aspect_ratio: (0.5, 2.0),
        };

        let components = make_components(
            &context_encoder,
            &target_encoder,
            &predictor,
            &energy_fn,
            &regularizer,
            &masking,
            0.0,
        );

        let input: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 4, 4], &device());
        let input_shape = InputShape::Image {
            height: 4,
            width: 4,
        };

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(123);
        let output = components.forward_step(&input, &input_shape, &mut rng);

        let energy_val: f32 = output.energy.value.into_scalar().elem();
        assert!(
            energy_val >= 0.0,
            "L2 energy should be >= 0, got {energy_val}"
        );
    }

    #[test]
    fn test_jepa_forward_step_predicted_shape() {
        let embed_dim = 16;
        let context_encoder = TestEncoder { embed_dim };
        let target_encoder = TestEncoder { embed_dim };
        let predictor = TestPredictor { embed_dim };
        let energy_fn = L2Energy;
        let regularizer = VICReg::default();
        let masking = BlockMasking {
            num_targets: 2,
            target_scale: (0.15, 0.3),
            target_aspect_ratio: (0.75, 1.5),
        };

        let batch = 3;
        let components = make_components(
            &context_encoder,
            &target_encoder,
            &predictor,
            &energy_fn,
            &regularizer,
            &masking,
            1.0,
        );

        let input: Tensor<TestBackend, 4> = Tensor::ones([batch, 1, 4, 4], &device());
        let input_shape = InputShape::Image {
            height: 4,
            width: 4,
        };

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(99);
        let output = components.forward_step(&input, &input_shape, &mut rng);

        assert_eq!(output.predicted.batch_size(), batch);
        assert_eq!(output.predicted.embed_dim(), embed_dim);
        assert_eq!(output.target.batch_size(), batch);
        assert_eq!(output.target.embed_dim(), embed_dim);
        assert_eq!(output.predicted.seq_len(), output.target.seq_len());
    }

    #[test]
    fn test_schedule_values() {
        use crate::schedule::WarmupCosineSchedule;

        let lr_schedule = WarmupCosineSchedule::new(1e-3, 1000, 10000);
        let ema = Ema::with_cosine_schedule(0.996, 10000);

        let (lr, momentum) = schedule_values(&lr_schedule, &ema, 0);
        assert!(lr.abs() < 1e-10, "lr at step 0 should be ~0");
        assert!(
            (momentum - 0.996).abs() < 1e-6,
            "momentum at step 0 should be ~0.996"
        );

        let (lr, momentum) = schedule_values(&lr_schedule, &ema, 1000);
        assert!((lr - 1e-3).abs() < 1e-6, "lr at warmup end should be peak");
        assert!(momentum > 0.996, "momentum should increase over training");
    }

    #[test]
    fn test_jepa_forward_step_with_reg_weight_zero() {
        let embed_dim = 8;
        let context_encoder = TestEncoder { embed_dim };
        let target_encoder = TestEncoder { embed_dim };
        let predictor = TestPredictor { embed_dim };
        let energy_fn = L2Energy;
        let regularizer = VICReg::default();
        let masking = BlockMasking {
            num_targets: 1,
            target_scale: (0.1, 0.3),
            target_aspect_ratio: (0.5, 2.0),
        };

        let components = make_components(
            &context_encoder,
            &target_encoder,
            &predictor,
            &energy_fn,
            &regularizer,
            &masking,
            0.0,
        );

        let input: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 4, 4], &device());
        let input_shape = InputShape::Image {
            height: 4,
            width: 4,
        };

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let output = components.forward_step(&input, &input_shape, &mut rng);

        // Total loss should equal energy when reg_weight is 0
        let energy_val: f32 = output.energy.value.into_scalar().elem();
        let total_val: f32 = output.total_loss.into_scalar().elem();
        assert!(
            (energy_val - total_val).abs() < 1e-6,
            "with reg_weight=0, total should equal energy: {energy_val} vs {total_val}"
        );
    }

    #[test]
    fn test_jepa_forward_step_with_cosine_energy() {
        use jepa_core::energy::CosineEnergy;

        let embed_dim = 16;
        let context_encoder = TestEncoder { embed_dim };
        let target_encoder = TestEncoder { embed_dim };
        let predictor = TestPredictor { embed_dim };
        let energy_fn = CosineEnergy;
        let regularizer = VICReg::default();
        let masking = BlockMasking {
            num_targets: 2,
            target_scale: (0.15, 0.3),
            target_aspect_ratio: (0.75, 1.5),
        };

        let components = JepaComponents::new(
            &context_encoder,
            &target_encoder,
            &predictor,
            &energy_fn,
            &regularizer,
            &masking,
            1.0,
        );

        let input: Tensor<TestBackend, 4> = Tensor::ones([2, 1, 4, 4], &device());
        let input_shape = InputShape::Image {
            height: 4,
            width: 4,
        };

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let output = components.forward_step(&input, &input_shape, &mut rng);

        let energy_val: f32 = output.energy.value.into_scalar().elem();
        assert!(
            energy_val.is_finite(),
            "cosine energy should be finite: {energy_val}"
        );

        let total_val: f32 = output.total_loss.into_scalar().elem();
        assert!(
            total_val.is_finite(),
            "total loss with cosine energy should be finite: {total_val}"
        );
    }

    #[test]
    fn test_jepa_forward_step_with_smooth_l1_energy() {
        use jepa_core::energy::SmoothL1Energy;

        let embed_dim = 16;
        let context_encoder = TestEncoder { embed_dim };
        let target_encoder = TestEncoder { embed_dim };
        let predictor = TestPredictor { embed_dim };
        let energy_fn = SmoothL1Energy::new(1.0);
        let regularizer = VICReg::default();
        let masking = BlockMasking {
            num_targets: 2,
            target_scale: (0.15, 0.3),
            target_aspect_ratio: (0.75, 1.5),
        };

        let components = JepaComponents::new(
            &context_encoder,
            &target_encoder,
            &predictor,
            &energy_fn,
            &regularizer,
            &masking,
            1.0,
        );

        let input: Tensor<TestBackend, 4> = Tensor::ones([2, 1, 4, 4], &device());
        let input_shape = InputShape::Image {
            height: 4,
            width: 4,
        };

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let output = components.forward_step(&input, &input_shape, &mut rng);

        let energy_val: f32 = output.energy.value.into_scalar().elem();
        assert!(
            energy_val.is_finite(),
            "smooth L1 energy should be finite: {energy_val}"
        );
        assert!(
            energy_val >= 0.0,
            "smooth L1 energy should be non-negative: {energy_val}"
        );
    }

    #[test]
    fn test_jepa_forward_step_with_barlow_twins_regularizer() {
        use jepa_core::collapse::BarlowTwins;

        let embed_dim = 16;
        let context_encoder = TestEncoder { embed_dim };
        let target_encoder = TestEncoder { embed_dim };
        let predictor = TestPredictor { embed_dim };
        let energy_fn = L2Energy;
        let regularizer = BarlowTwins::default();
        let masking = BlockMasking {
            num_targets: 2,
            target_scale: (0.15, 0.3),
            target_aspect_ratio: (0.75, 1.5),
        };

        let components = JepaComponents::new(
            &context_encoder,
            &target_encoder,
            &predictor,
            &energy_fn,
            &regularizer,
            &masking,
            1.0,
        );

        let input: Tensor<TestBackend, 4> = Tensor::ones([2, 1, 4, 4], &device());
        let input_shape = InputShape::Image {
            height: 4,
            width: 4,
        };

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let output = components.forward_step(&input, &input_shape, &mut rng);

        let total_val: f32 = output.total_loss.into_scalar().elem();
        assert!(
            total_val.is_finite(),
            "total loss with Barlow Twins should be finite: {total_val}"
        );
    }
}
