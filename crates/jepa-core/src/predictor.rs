//! Predictor trait for JEPA.
//!
//! The predictor is the *generative* component of JEPA. Given the context
//! encoder's output and a set of target positions, it predicts what the
//! target encoder would produce for those positions.
//!
//! ```text
//! context representation ──┐
//!                          ▼
//!                    ┌────────────┐
//!    target_pos ────►│  Predictor │──► predicted target representation
//!    z (optional) ──►│            │
//!                    └────────────┘
//! ```
//!
//! In I-JEPA and V-JEPA the predictor is a narrow transformer that
//! cross-attends from learnable prediction tokens to the context. The
//! optional latent variable `z` supports stochastic (energy-based)
//! prediction, where the model captures multi-modal uncertainty.
//!
//! See [`crate::energy`] for functions that measure prediction quality.

use burn::tensor::{backend::Backend, Tensor};

use crate::types::Representation;

/// Trait for JEPA predictors.
///
/// A predictor takes the context encoder's output and predicts the
/// target encoder's output at specified positions. This is the core
/// generative component of JEPA — the only part that is trained with
/// gradients in the standard JEPA loop.
///
/// Concrete implementations:
/// - [`jepa_vision::TransformerPredictor`](../../jepa_vision/image/struct.TransformerPredictor.html) — narrow cross-attention predictor (I-JEPA / V-JEPA)
///
/// # Type parameters
///
/// - `B` — burn backend
pub trait Predictor<B: Backend> {
    /// Predict target representation from context representation.
    ///
    /// # Arguments
    /// * `context` - Representation from the context encoder. Shape: `[batch, ctx_seq_len, embed_dim]`
    /// * `target_positions` - Positions of the target tokens to predict. Shape: `[batch, num_targets]`
    /// * `latent` - Optional latent variable z for stochastic predictions. Shape: `[batch, latent_dim]`
    ///
    /// # Returns
    /// Predicted representation at the target positions. Shape: `[batch, num_targets, embed_dim]`
    fn predict(
        &self,
        context: &Representation<B>,
        target_positions: &Tensor<B, 2>,
        latent: Option<&Tensor<B, 2>>,
    ) -> Representation<B>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    /// A trivial predictor that returns zeros for testing.
    struct ZeroPredictor {
        embed_dim: usize,
    }

    impl Predictor<TestBackend> for ZeroPredictor {
        fn predict(
            &self,
            _context: &Representation<TestBackend>,
            target_positions: &Tensor<TestBackend, 2>,
            _latent: Option<&Tensor<TestBackend, 2>>,
        ) -> Representation<TestBackend> {
            let [batch, num_targets] = target_positions.dims();
            let device = target_positions.device();
            Representation::new(Tensor::zeros([batch, num_targets, self.embed_dim], &device))
        }
    }

    #[test]
    fn test_predictor_trait_is_implementable() {
        let predictor = ZeroPredictor { embed_dim: 64 };
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let context = Representation::new(Tensor::zeros([2, 8, 64], &device));
        let target_pos: Tensor<TestBackend, 2> = Tensor::zeros([2, 4], &device);
        let predicted = predictor.predict(&context, &target_pos, None);
        assert_eq!(predicted.batch_size(), 2);
        assert_eq!(predicted.seq_len(), 4);
        assert_eq!(predicted.embed_dim(), 64);
    }

    #[test]
    fn test_predictor_with_latent() {
        let predictor = ZeroPredictor { embed_dim: 64 };
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let context = Representation::new(Tensor::zeros([2, 8, 64], &device));
        let target_pos: Tensor<TestBackend, 2> = Tensor::zeros([2, 4], &device);
        let latent: Tensor<TestBackend, 2> = Tensor::zeros([2, 16], &device);
        let predicted = predictor.predict(&context, &target_pos, Some(&latent));
        assert_eq!(predicted.batch_size(), 2);
        assert_eq!(predicted.seq_len(), 4);
    }
}
