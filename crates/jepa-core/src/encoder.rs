//! Encoder trait for JEPA.
//!
//! Implements RFC-002 (Encoder Module).
//!
//! An encoder maps raw input (images, video, or already-embedded tokens)
//! into a [`Representation`] in embedding space.
//!
//! In a JEPA training loop **two** encoder instances exist:
//!
//! | Role | Gradients? | Weight update |
//! |------|-----------|---------------|
//! | **Context encoder** (θ) | Yes | Backpropagation |
//! | **Target encoder** (ξ) | No | EMA of θ (see [`crate::ema::Ema`]) |
//!
//! Both share the same architecture and implement this trait. The
//! asymmetric update (EMA on the target) is what prevents collapse.
//!
//! See [`crate::collapse`] for the regularizers that complement EMA.

use burn::tensor::backend::Backend;

use crate::types::Representation;

/// Trait for JEPA encoders.
///
/// An encoder maps raw input to a [`Representation`] with shape
/// `[batch, seq_len, embed_dim]`. Concrete implementations include:
///
/// - [`jepa_vision::VitEncoder`](../../jepa_vision/vit/struct.VitEncoder.html) — Vision Transformer for images
/// - [`jepa_vision::VitVideoEncoder`](../../jepa_vision/video/struct.VitVideoEncoder.html) — Vision Transformer for video
///
/// # Type parameters
///
/// - `B` — burn backend (e.g. `NdArray`, `Wgpu`, `Tch`)
///
/// # Associated types
///
/// - `Input` — the raw input type this encoder accepts. For vision
///   encoders this is typically a `Tensor<B, 4>` (images) or
///   `Tensor<B, 5>` (video). Higher-level wrappers may accept
///   [`Representation<B>`] so that levels in a hierarchy can chain.
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
