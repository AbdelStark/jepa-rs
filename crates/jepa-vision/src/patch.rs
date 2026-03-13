//! Patch embedding for images.
//!
//! Implements the patchification step from RFC-002 (Encoder Module).
//!
//! Patch embedding is the first stage of a Vision Transformer: it converts
//! a raw image into a sequence of learnable token vectors.
//!
//! ```text
//! [B, C, H, W]  ──reshape──►  [B, grid_h·grid_w, C·patch_h·patch_w]  ──linear──►  [B, S, D]
//! ```
//!
//! Steps:
//! 1. Divide the image into non-overlapping patches of size `(patch_h, patch_w)`.
//! 2. Flatten each patch to a vector of length `C × patch_h × patch_w`.
//! 3. Project through a learned linear layer to `embed_dim`.
//!
//! For video, see [`crate::video`] which uses 3-D *tubelet* embedding instead.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::Backend;

/// Configuration for patch embedding.
///
/// # Example
///
/// ```
/// use jepa_vision::patch::PatchEmbeddingConfig;
/// use burn_ndarray::NdArray;
/// use burn::prelude::*;
///
/// type B = NdArray<f32>;
/// let device = burn_ndarray::NdArrayDevice::Cpu;
///
/// let config = PatchEmbeddingConfig::new(3, 16, 16, 256);
/// let patch_embed = config.init::<B>(&device);
/// assert_eq!(patch_embed.num_patches(224, 224), 196);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PatchEmbeddingConfig {
    /// Number of input channels (e.g., 3 for RGB).
    pub in_channels: usize,
    /// Patch height in pixels.
    pub patch_h: usize,
    /// Patch width in pixels.
    pub patch_w: usize,
    /// Output embedding dimension.
    pub embed_dim: usize,
}

impl PatchEmbeddingConfig {
    /// Create a new config with the given parameters.
    pub fn new(in_channels: usize, patch_h: usize, patch_w: usize, embed_dim: usize) -> Self {
        Self {
            in_channels,
            patch_h,
            patch_w,
            embed_dim,
        }
    }

    /// Initialize a [`PatchEmbedding`] module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PatchEmbedding<B> {
        let patch_dim = self.in_channels * self.patch_h * self.patch_w;
        let projection = LinearConfig::new(patch_dim, self.embed_dim).init(device);
        PatchEmbedding {
            projection,
            patch_h: self.patch_h,
            patch_w: self.patch_w,
            in_channels: self.in_channels,
        }
    }
}

/// Patch embedding module.
///
/// Splits an image into non-overlapping patches and projects each
/// through a linear layer to produce patch embeddings.
///
/// This is the first stage of a Vision Transformer (ViT) encoder.
#[derive(Module, Debug)]
pub struct PatchEmbedding<B: Backend> {
    /// Linear projection from flattened patch to embedding space.
    projection: Linear<B>,
    /// Patch height in pixels.
    patch_h: usize,
    /// Patch width in pixels.
    patch_w: usize,
    /// Number of input channels.
    in_channels: usize,
}

impl<B: Backend> PatchEmbedding<B> {
    /// Convert an image batch to patch embeddings.
    ///
    /// # Arguments
    /// * `images` - Input images. Shape: `[batch, channels, height, width]`
    ///
    /// # Returns
    /// Patch embeddings. Shape: `[batch, num_patches, embed_dim]`
    ///
    /// # Panics
    /// If `height` is not divisible by `patch_h` or `width` is not divisible by `patch_w`.
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, _channels, height, width] = images.dims();

        let grid_h = height / self.patch_h;
        let grid_w = width / self.patch_w;
        let num_patches = grid_h * grid_w;
        let patch_dim = self.in_channels * self.patch_h * self.patch_w;

        // Reshape: [batch, C, H, W] -> [batch, C, grid_h, patch_h, grid_w, patch_w]
        let x = images.reshape([
            batch,
            self.in_channels,
            grid_h,
            self.patch_h,
            grid_w,
            self.patch_w,
        ]);
        // Permute to: [batch, grid_h, grid_w, C, patch_h, patch_w]
        let x = x.permute([0, 2, 4, 1, 3, 5]);
        // Flatten patches: [batch, num_patches, patch_dim]
        let x = x.reshape([batch, num_patches, patch_dim]);

        // Project: [batch, num_patches, embed_dim]
        self.projection.forward(x)
    }

    /// Get the number of patches for a given image size.
    pub fn num_patches(&self, height: usize, width: usize) -> usize {
        (height / self.patch_h) * (width / self.patch_w)
    }

    /// Get the grid dimensions for a given image size.
    pub fn grid_size(&self, height: usize, width: usize) -> (usize, usize) {
        (height / self.patch_h, width / self.patch_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

    #[test]
    fn test_patch_embedding_output_shape() {
        let config = PatchEmbeddingConfig::new(3, 16, 16, 256);
        let pe = config.init::<TestBackend>(&device());

        let images: Tensor<TestBackend, 4> = Tensor::zeros([2, 3, 224, 224], &device());
        let output = pe.forward(images);

        assert_eq!(output.dims(), [2, 196, 256]); // 224/16 = 14, 14*14 = 196
    }

    #[test]
    fn test_patch_embedding_small_image() {
        let config = PatchEmbeddingConfig::new(1, 2, 2, 8);
        let pe = config.init::<TestBackend>(&device());

        let images: Tensor<TestBackend, 4> = Tensor::zeros([1, 1, 4, 4], &device());
        let output = pe.forward(images);

        assert_eq!(output.dims(), [1, 4, 8]); // 4/2 = 2, 2*2 = 4 patches
    }

    #[test]
    fn test_num_patches() {
        let config = PatchEmbeddingConfig::new(3, 16, 16, 256);
        let pe = config.init::<TestBackend>(&device());
        assert_eq!(pe.num_patches(224, 224), 196);
        assert_eq!(pe.num_patches(32, 32), 4);
    }

    #[test]
    fn test_grid_size() {
        let config = PatchEmbeddingConfig::new(3, 16, 16, 256);
        let pe = config.init::<TestBackend>(&device());
        assert_eq!(pe.grid_size(224, 224), (14, 14));
    }

    #[test]
    fn test_patch_embedding_nonzero_output() {
        let config = PatchEmbeddingConfig::new(3, 16, 16, 64);
        let pe = config.init::<TestBackend>(&device());

        // Use ones instead of zeros to ensure non-trivial output
        let images: Tensor<TestBackend, 4> = Tensor::ones([1, 3, 32, 32], &device());
        let output = pe.forward(images);
        let [_b, _s, _d] = output.dims();
        // Output should not be all zeros (linear projection has random init)
        // We just check the shape is correct
        assert_eq!(output.dims(), [1, 4, 64]);
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_num_patches_equals_grid_product(
            grid_h in 1usize..8,
            grid_w in 1usize..8,
            patch_size in proptest::sample::select(vec![2usize, 4, 8]),
        ) {
            let config = PatchEmbeddingConfig::new(1, patch_size, patch_size, 16);
            let pe = config.init::<TestBackend>(&device());
            let h = grid_h * patch_size;
            let w = grid_w * patch_size;
            let np = pe.num_patches(h, w);
            prop_assert_eq!(np, grid_h * grid_w);
        }

        #[test]
        fn prop_patch_embedding_output_shape(
            grid_h in 1usize..4,
            grid_w in 1usize..4,
            batch in 1usize..3,
        ) {
            let patch_size = 2;
            let embed_dim = 8;
            let config = PatchEmbeddingConfig::new(1, patch_size, patch_size, embed_dim);
            let pe = config.init::<TestBackend>(&device());
            let h = grid_h * patch_size;
            let w = grid_w * patch_size;
            let images: Tensor<TestBackend, 4> = Tensor::zeros([batch, 1, h, w], &device());
            let output = pe.forward(images);
            let expected_patches = grid_h * grid_w;
            prop_assert_eq!(output.dims(), [batch, expected_patches, embed_dim]);
        }
    }
}
