//! Vision Transformer (ViT) encoder for JEPA.
//!
//! Implements RFC-002 (Encoder Module) — concrete ViT encoder.
//!
//! The Vision Transformer encodes images into patch-level representations
//! by applying self-attention transformer blocks over patch embeddings.
//!
//! Architecture:
//! 1. Patch embedding: image → patch sequence
//! 2. Position encoding: 2D RoPE
//! 3. Transformer blocks: self-attention + MLP
//! 4. Layer normalization

use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{backend::Backend, Int, TensorData};

use jepa_core::types::Representation;
use jepa_core::Encoder;

use crate::patch::{PatchEmbedding, PatchEmbeddingConfig};
use crate::rope::{RotaryPositionEncoding2D, RotaryPositionEncoding2DConfig};

/// Configuration for a Vision Transformer encoder.
///
/// # Example
///
/// ```
/// use jepa_vision::vit::VitConfig;
/// use jepa_core::Encoder;
/// use burn_ndarray::NdArray;
///
/// type B = NdArray<f32>;
/// let device = burn_ndarray::NdArrayDevice::Cpu;
///
/// let config = VitConfig::tiny_test();
/// let encoder = config.init::<B>(&device);
/// assert_eq!(encoder.embed_dim(), 32);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VitConfig {
    /// Number of input channels (e.g., 3 for RGB).
    pub in_channels: usize,
    /// Input image height in pixels.
    pub image_height: usize,
    /// Input image width in pixels.
    pub image_width: usize,
    /// Patch size `(height, width)`.
    pub patch_size: (usize, usize),
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// MLP hidden dimension (typically 4 * embed_dim).
    pub mlp_dim: usize,
    /// Dropout rate (not used during inference).
    pub dropout: f64,
}

impl VitConfig {
    /// Create a ViT-Base/16 config for 224x224 images.
    pub fn vit_base_patch16() -> Self {
        Self {
            in_channels: 3,
            image_height: 224,
            image_width: 224,
            patch_size: (16, 16),
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_dim: 3072,
            dropout: 0.0,
        }
    }

    /// Create a ViT-Small/16 config for 224x224 images.
    pub fn vit_small_patch16() -> Self {
        Self {
            in_channels: 3,
            image_height: 224,
            image_width: 224,
            patch_size: (16, 16),
            embed_dim: 384,
            num_layers: 12,
            num_heads: 6,
            mlp_dim: 1536,
            dropout: 0.0,
        }
    }

    /// Create a ViT-Large/16 config for 224x224 images.
    ///
    /// Matches the architecture used in Facebook Research I-JEPA ViT-L/16.
    pub fn vit_large_patch16() -> Self {
        Self {
            in_channels: 3,
            image_height: 224,
            image_width: 224,
            patch_size: (16, 16),
            embed_dim: 1024,
            num_layers: 24,
            num_heads: 16,
            mlp_dim: 4096,
            dropout: 0.0,
        }
    }

    /// Create a ViT-Huge/14 config for 224x224 images.
    ///
    /// Matches the architecture used in Facebook Research I-JEPA ViT-H/14
    /// (the primary model released with the I-JEPA paper).
    pub fn vit_huge_patch14() -> Self {
        Self {
            in_channels: 3,
            image_height: 224,
            image_width: 224,
            patch_size: (14, 14),
            embed_dim: 1280,
            num_layers: 32,
            num_heads: 16,
            mlp_dim: 5120,
            dropout: 0.0,
        }
    }

    /// Create a ViT-Huge/16 config for 448x448 images.
    ///
    /// Matches the architecture used in Facebook Research I-JEPA ViT-H/16-448.
    pub fn vit_huge_patch16_448() -> Self {
        Self {
            in_channels: 3,
            image_height: 448,
            image_width: 448,
            patch_size: (16, 16),
            embed_dim: 1280,
            num_layers: 32,
            num_heads: 16,
            mlp_dim: 5120,
            dropout: 0.0,
        }
    }

    /// Create a ViT-Giant/16 config for 224x224 images.
    ///
    /// Matches the architecture used in Facebook Research I-JEPA ViT-G/16.
    pub fn vit_giant_patch16() -> Self {
        Self {
            in_channels: 3,
            image_height: 224,
            image_width: 224,
            patch_size: (16, 16),
            embed_dim: 1408,
            num_layers: 40,
            num_heads: 16,
            mlp_dim: 6144,
            dropout: 0.0,
        }
    }

    /// Create a minimal config for testing.
    pub fn tiny_test() -> Self {
        Self {
            in_channels: 1,
            image_height: 8,
            image_width: 8,
            patch_size: (2, 2),
            embed_dim: 32,
            num_layers: 2,
            num_heads: 4,
            mlp_dim: 64,
            dropout: 0.0,
        }
    }

    fn grid_height(&self) -> usize {
        self.image_height / self.patch_size.0
    }

    fn grid_width(&self) -> usize {
        self.image_width / self.patch_size.1
    }

    /// Initialize a [`VitEncoder`] module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> VitEncoder<B> {
        let patch_embed_config = PatchEmbeddingConfig::new(
            self.in_channels,
            self.patch_size.0,
            self.patch_size.1,
            self.embed_dim,
        );
        let patch_embed = patch_embed_config.init(device);

        let rope_config = RotaryPositionEncoding2DConfig::new(
            self.embed_dim,
            self.grid_height(),
            self.grid_width(),
        );
        let positional_encoding = rope_config.init(device);

        let blocks: Vec<TransformerBlock<B>> = (0..self.num_layers)
            .map(|_| {
                TransformerBlockConfig {
                    embed_dim: self.embed_dim,
                    num_heads: self.num_heads,
                    mlp_dim: self.mlp_dim,
                }
                .init(device)
            })
            .collect();

        let norm = LayerNormConfig::new(self.embed_dim).init(device);

        VitEncoder {
            patch_embed,
            positional_encoding,
            blocks,
            norm,
            embed_dim: self.embed_dim,
        }
    }
}

/// Vision Transformer encoder.
///
/// Maps images to patch-level representations via:
/// 1. Patch embedding (linear projection of flattened patches)
/// 2. 2D Rotary Position Encoding
/// 3. Stack of transformer blocks (self-attention + MLP)
/// 4. Final layer normalization
///
/// Output shape: `[batch, num_patches, embed_dim]`
#[derive(Module, Debug)]
pub struct VitEncoder<B: Backend> {
    /// Patch embedding: image → patch tokens.
    patch_embed: PatchEmbedding<B>,
    /// 2D Rotary Position Encoding.
    positional_encoding: RotaryPositionEncoding2D<B>,
    /// Stack of transformer blocks.
    blocks: Vec<TransformerBlock<B>>,
    /// Final layer normalization.
    norm: LayerNorm<B>,
    /// Output embedding dimension.
    embed_dim: usize,
}

impl<B: Backend> VitEncoder<B> {
    fn positioned_patch_tokens(&self, images: &Tensor<B, 4>) -> Tensor<B, 3> {
        // 1. Patch embedding
        let x = self.patch_embed.forward(images.clone());

        // 2. Apply RoPE before any masking so absolute token positions remain correct.
        self.positional_encoding.forward(x)
    }

    fn encode_positioned_tokens(&self, mut x: Tensor<B, 3>) -> Representation<B> {
        // Transformer blocks
        for block in &self.blocks {
            x = block.forward(x);
        }

        // Layer norm
        x = self.norm.forward(x);

        Representation::new(x)
    }

    /// Forward pass: image → representation.
    ///
    /// # Arguments
    /// * `images` - Input images. Shape: `[batch, channels, height, width]`
    ///
    /// # Returns
    /// Patch-level representations. Shape: `[batch, num_patches, embed_dim]`
    pub fn forward(&self, images: &Tensor<B, 4>) -> Representation<B> {
        let x = self.positioned_patch_tokens(images);
        self.encode_positioned_tokens(x)
    }

    /// Encode only the visible patch tokens for strict JEPA context encoding.
    ///
    /// The image is patchified and position-encoded using the full grid so the
    /// surviving tokens retain their real flattened positions, then masked
    /// tokens are removed before self-attention runs.
    pub fn forward_visible_tokens(
        &self,
        images: &Tensor<B, 4>,
        visible_indices: &[usize],
    ) -> Representation<B> {
        let x = self.positioned_patch_tokens(images);
        let x = gather_token_sequence(x, visible_indices);
        self.encode_positioned_tokens(x)
    }
}

fn gather_token_sequence<B: Backend>(tokens: Tensor<B, 3>, indices: &[usize]) -> Tensor<B, 3> {
    let [batch, seq_len, embed_dim] = tokens.dims();
    let device = tokens.device();

    if indices.is_empty() {
        return Tensor::zeros([batch, 0, embed_dim], &device);
    }

    // Validate that all indices are within bounds before calling select(),
    // which may panic or produce undefined results on out-of-range indices.
    for &idx in indices {
        assert!(
            idx < seq_len,
            "gather index {idx} out of bounds for sequence length {seq_len}",
        );
    }

    let index_data: Vec<i64> = indices.iter().map(|&index| index as i64).collect();
    let index_tensor =
        Tensor::<B, 1, Int>::from_data(TensorData::new(index_data, [indices.len()]), &device);

    tokens.select(1, index_tensor)
}

impl<B: Backend> Encoder<B> for VitEncoder<B> {
    type Input = Tensor<B, 4>;

    fn encode(&self, input: &Self::Input) -> Representation<B> {
        self.forward(input)
    }

    fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

// --- Transformer components ---

/// Configuration for a transformer block.
#[derive(Debug, Clone)]
struct TransformerBlockConfig {
    embed_dim: usize,
    num_heads: usize,
    mlp_dim: usize,
}

impl TransformerBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        TransformerBlock {
            norm1: LayerNormConfig::new(self.embed_dim).init(device),
            attn: MultiHeadSelfAttentionConfig {
                embed_dim: self.embed_dim,
                num_heads: self.num_heads,
            }
            .init(device),
            norm2: LayerNormConfig::new(self.embed_dim).init(device),
            mlp: MlpConfig {
                in_dim: self.embed_dim,
                hidden_dim: self.mlp_dim,
            }
            .init(device),
        }
    }
}

/// Pre-norm transformer block: LN → Attention → residual → LN → MLP → residual.
#[derive(Module, Debug)]
struct TransformerBlock<B: Backend> {
    norm1: LayerNorm<B>,
    attn: MultiHeadSelfAttention<B>,
    norm2: LayerNorm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> TransformerBlock<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-norm attention with residual
        let residual = x.clone();
        let x_norm = self.norm1.forward(x);
        let attn_out = self.attn.forward(x_norm);
        let x = residual + attn_out;

        // Pre-norm MLP with residual
        let residual = x.clone();
        let x_norm = self.norm2.forward(x);
        let mlp_out = self.mlp.forward(x_norm);
        residual + mlp_out
    }
}

// --- Multi-Head Self-Attention ---

#[derive(Debug, Clone)]
struct MultiHeadSelfAttentionConfig {
    embed_dim: usize,
    num_heads: usize,
}

impl MultiHeadSelfAttentionConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadSelfAttention<B> {
        let head_dim = self.embed_dim / self.num_heads;
        MultiHeadSelfAttention {
            qkv: LinearConfig::new(self.embed_dim, 3 * self.embed_dim).init(device),
            out_proj: LinearConfig::new(self.embed_dim, self.embed_dim).init(device),
            num_heads: self.num_heads,
            head_dim,
        }
    }
}

/// Multi-head self-attention.
///
/// Computes scaled dot-product attention across multiple heads.
#[derive(Module, Debug)]
struct MultiHeadSelfAttention<B: Backend> {
    /// Combined QKV projection.
    qkv: Linear<B>,
    /// Output projection.
    out_proj: Linear<B>,
    /// Number of attention heads.
    num_heads: usize,
    /// Dimension per head.
    head_dim: usize,
}

impl<B: Backend> MultiHeadSelfAttention<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _embed_dim] = x.dims();
        let embed_dim = self.num_heads * self.head_dim;

        // Combined QKV: [batch, seq_len, 3 * embed_dim]
        let qkv = self.qkv.forward(x);

        // Split into Q, K, V
        let q = qkv.clone().slice([0..batch, 0..seq_len, 0..embed_dim]);
        let k = qkv
            .clone()
            .slice([0..batch, 0..seq_len, embed_dim..2 * embed_dim]);
        let v = qkv.slice([0..batch, 0..seq_len, 2 * embed_dim..3 * embed_dim]);

        // Reshape to multi-head: [batch, seq_len, num_heads, head_dim] → [batch, num_heads, seq_len, head_dim]
        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(k.transpose()) / scale; // [batch, heads, seq, seq]
        let attn_weights = burn::tensor::activation::softmax(attn_weights, 3);

        // Apply attention to values
        let out = attn_weights.matmul(v); // [batch, heads, seq, head_dim]

        // Reshape back: [batch, seq_len, embed_dim]
        let out = out.swap_dims(1, 2).reshape([batch, seq_len, embed_dim]);

        self.out_proj.forward(out)
    }
}

// --- MLP ---

#[derive(Debug, Clone)]
struct MlpConfig {
    in_dim: usize,
    hidden_dim: usize,
}

impl MlpConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        Mlp {
            fc1: LinearConfig::new(self.in_dim, self.hidden_dim).init(device),
            fc2: LinearConfig::new(self.hidden_dim, self.in_dim).init(device),
        }
    }
}

/// Two-layer MLP with GELU activation.
#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> Mlp<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = burn::tensor::activation::gelu(x);
        self.fc2.forward(x)
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
    fn test_vit_encoder_output_shape() {
        let config = VitConfig::tiny_test();
        let encoder = config.init::<TestBackend>(&device());

        let images: Tensor<TestBackend, 4> = Tensor::zeros([2, 1, 8, 8], &device());
        let repr = encoder.forward(&images);

        // 8/2 = 4 patches per side, 4*4 = 16 patches total
        assert_eq!(repr.batch_size(), 2);
        assert_eq!(repr.seq_len(), 16);
        assert_eq!(repr.embed_dim(), 32);
    }

    #[test]
    fn test_vit_encoder_trait_impl() {
        let config = VitConfig::tiny_test();
        let encoder = config.init::<TestBackend>(&device());

        let images: Tensor<TestBackend, 4> = Tensor::zeros([1, 1, 8, 8], &device());
        let repr = Encoder::encode(&encoder, &images);

        assert_eq!(repr.batch_size(), 1);
        assert_eq!(repr.seq_len(), 16);
        assert_eq!(encoder.embed_dim(), 32);
    }

    #[test]
    fn test_vit_encoder_different_inputs_different_outputs() {
        let config = VitConfig::tiny_test();
        let encoder = config.init::<TestBackend>(&device());

        let a: Tensor<TestBackend, 4> = Tensor::zeros([1, 1, 8, 8], &device());
        let b: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 8, 8], &device());

        let repr_a = encoder.forward(&a);
        let repr_b = encoder.forward(&b);

        let diff: f32 = (repr_a.embeddings - repr_b.embeddings)
            .abs()
            .sum()
            .into_scalar()
            .elem();
        assert!(
            diff > 1e-6,
            "different inputs should produce different representations"
        );
    }

    #[test]
    fn test_transformer_block_residual() {
        // Verify the transformer block preserves the residual connection
        let block = TransformerBlockConfig {
            embed_dim: 16,
            num_heads: 2,
            mlp_dim: 32,
        }
        .init::<TestBackend>(&device());

        let x: Tensor<TestBackend, 3> = Tensor::zeros([1, 4, 16], &device());
        let out = block.forward(x);
        assert_eq!(out.dims(), [1, 4, 16]);
    }

    #[test]
    fn test_mhsa_output_shape() {
        let attn = MultiHeadSelfAttentionConfig {
            embed_dim: 16,
            num_heads: 4,
        }
        .init::<TestBackend>(&device());

        let x: Tensor<TestBackend, 3> = Tensor::zeros([2, 8, 16], &device());
        let out = attn.forward(x);
        assert_eq!(out.dims(), [2, 8, 16]);
    }

    #[test]
    fn test_mlp_output_shape() {
        let mlp = MlpConfig {
            in_dim: 16,
            hidden_dim: 64,
        }
        .init::<TestBackend>(&device());

        let x: Tensor<TestBackend, 3> = Tensor::zeros([2, 8, 16], &device());
        let out = mlp.forward(x);
        assert_eq!(out.dims(), [2, 8, 16]);
    }

    use burn::tensor::ElementConversion;
    use proptest::prelude::*;

    proptest! {
        /// Property: ViT encoder output is always finite (no NaN/Inf) for
        /// small normally-distributed inputs.
        #[test]
        fn prop_vit_output_is_finite(batch in 1usize..3) {
            let config = VitConfig::tiny_test();
            let encoder = config.init::<TestBackend>(&device());

            let images: Tensor<TestBackend, 4> = Tensor::random(
                [batch, 1, 8, 8],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device(),
            );
            let repr = encoder.forward(&images);

            // Check shape
            prop_assert_eq!(repr.batch_size(), batch);
            prop_assert_eq!(repr.seq_len(), 16);
            prop_assert_eq!(repr.embed_dim(), 32);

            // Check finiteness: sum of abs should be finite and non-NaN
            let total: f32 = repr.embeddings.abs().sum().into_scalar().elem();
            prop_assert!(total.is_finite(), "ViT output should be finite, got {}", total);
        }

        /// Property: ViT encoder is deterministic — same input always produces
        /// the same output.
        #[test]
        fn prop_vit_is_deterministic(batch in 1usize..3) {
            let config = VitConfig::tiny_test();
            let encoder = config.init::<TestBackend>(&device());

            let images: Tensor<TestBackend, 4> = Tensor::ones([batch, 1, 8, 8], &device());
            let repr1 = encoder.forward(&images);
            let repr2 = encoder.forward(&images);

            let diff: f32 = (repr1.embeddings - repr2.embeddings)
                .abs()
                .sum()
                .into_scalar()
                .elem();
            prop_assert!(diff < 1e-6, "ViT should be deterministic, diff={}", diff);
        }

        /// Property: transformer block preserves tensor dimensions for any
        /// valid (embed_dim, num_heads) combination where embed_dim % num_heads == 0.
        #[test]
        fn prop_transformer_block_preserves_shape(
            seq_len in 2usize..8,
            num_heads in proptest::sample::select(vec![2usize, 4]),
        ) {
            let embed_dim = 16; // divisible by 2 and 4
            let block = TransformerBlockConfig {
                embed_dim,
                num_heads,
                mlp_dim: embed_dim * 4,
            }
            .init::<TestBackend>(&device());

            let x: Tensor<TestBackend, 3> = Tensor::random(
                [1, seq_len, embed_dim],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device(),
            );
            let out = block.forward(x);
            prop_assert_eq!(out.dims(), [1, seq_len, embed_dim]);

            let total: f32 = out.abs().sum().into_scalar().elem();
            prop_assert!(total.is_finite(), "block output should be finite");
        }
    }
}
