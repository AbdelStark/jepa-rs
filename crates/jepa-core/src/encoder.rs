//! Encoder trait for JEPA.
//!
//! Implements RFC-002 (Encoder Module) — core trait definition.
//!
//! An encoder maps raw input to a representation in embedding space.
//! In JEPA, both the context encoder and target encoder implement this trait.
//! The target encoder's weights are updated via EMA (RFC-007), not gradients.

use burn::tensor::backend::Backend;

use crate::types::Representation;

/// Trait for JEPA encoders.
///
/// An encoder maps raw input to a [`Representation`]. Implementations
/// include Vision Transformers (ViT) for images and video.
///
/// # Type Parameters
/// * `B` - The burn backend (e.g., NdArray, Wgpu)
///
/// # Associated Types
/// * `Input` - The type of raw input the encoder accepts
pub trait Encoder<B: Backend> {
    /// The type of input this encoder accepts.
    type Input;

    /// Encode input into a representation.
    ///
    /// # Arguments
    /// * `input` - The raw input to encode
    ///
    /// # Returns
    /// A [`Representation`] with shape `[batch, seq_len, embed_dim]`
    fn encode(&self, input: &Self::Input) -> Representation<B>;

    /// Get the output embedding dimension.
    fn embed_dim(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    /// A trivial encoder for testing the trait definition.
    struct IdentityEncoder {
        dim: usize,
    }

    impl Encoder<TestBackend> for IdentityEncoder {
        type Input = Tensor<TestBackend, 3>;

        fn encode(&self, input: &Self::Input) -> Representation<TestBackend> {
            Representation::new(input.clone())
        }

        fn embed_dim(&self) -> usize {
            self.dim
        }
    }

    #[test]
    fn test_encoder_trait_is_implementable() {
        let encoder = IdentityEncoder { dim: 64 };
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let input: Tensor<TestBackend, 3> = Tensor::zeros([2, 8, 64], &device);
        let repr = encoder.encode(&input);
        assert_eq!(repr.batch_size(), 2);
        assert_eq!(repr.seq_len(), 8);
        assert_eq!(repr.embed_dim(), 64);
        assert_eq!(encoder.embed_dim(), 64);
    }
}
