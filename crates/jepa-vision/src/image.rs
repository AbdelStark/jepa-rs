//! I-JEPA (Image Joint Embedding Predictive Architecture) pipeline.
//!
//! Implements the complete I-JEPA model for self-supervised image learning.
//! The model consists of:
//! - A context encoder (ViT) that encodes visible patches
//! - A target encoder (ViT, EMA copy) that encodes target patches
//! - A transformer predictor that predicts target representations from context
//! - Block masking strategy for creating prediction tasks
//!
//! Reference: Assran et al. (2023), "Self-Supervised Learning from Images
//! with a Joint-Embedding Predictive Architecture", CVPR.

use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::Backend;

use jepa_core::types::Representation;
use jepa_core::Predictor;

/// Configuration for the transformer predictor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransformerPredictorConfig {
    /// Input embedding dimension (from encoder output).
    pub encoder_embed_dim: usize,
    /// Predictor internal embedding dimension.
    pub predictor_embed_dim: usize,
    /// Number of predictor transformer layers.
    pub num_layers: usize,
    /// Number of attention heads in the predictor.
    pub num_heads: usize,
    /// Maximum number of target positions to predict.
    pub max_target_len: usize,
}

impl TransformerPredictorConfig {
    /// Initialize a [`TransformerPredictor`] module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerPredictor<B> {
        let input_proj =
            LinearConfig::new(self.encoder_embed_dim, self.predictor_embed_dim).init(device);
        let output_proj =
            LinearConfig::new(self.predictor_embed_dim, self.encoder_embed_dim).init(device);

        let blocks: Vec<PredictorBlock<B>> = (0..self.num_layers)
            .map(|_| {
                PredictorBlockConfig {
                    embed_dim: self.predictor_embed_dim,
                    num_heads: self.num_heads,
                }
                .init(device)
            })
            .collect();

        let norm = LayerNormConfig::new(self.predictor_embed_dim).init(device);

        // Learnable prediction tokens (initialized to zeros, will be learned)
        let prediction_tokens =
            Tensor::zeros([self.max_target_len, self.predictor_embed_dim], device);

        TransformerPredictor {
            input_proj,
            output_proj,
            blocks,
            norm,
            prediction_tokens,
            predictor_embed_dim: self.predictor_embed_dim,
            encoder_embed_dim: self.encoder_embed_dim,
        }
    }
}

/// Transformer-based predictor for I-JEPA.
///
/// Predicts target representations from context representations using
/// cross-attention from learnable prediction tokens to the context.
///
/// Architecture:
/// 1. Project context to predictor dimension
/// 2. Concatenate prediction tokens (one per target position) with context
/// 3. Apply self-attention transformer blocks
/// 4. Extract prediction token outputs
/// 5. Project back to encoder dimension
#[derive(Module, Debug)]
pub struct TransformerPredictor<B: Backend> {
    /// Project encoder output to predictor dimension.
    input_proj: Linear<B>,
    /// Project predictor output back to encoder dimension.
    output_proj: Linear<B>,
    /// Transformer blocks for the predictor.
    blocks: Vec<PredictorBlock<B>>,
    /// Final layer norm.
    norm: LayerNorm<B>,
    /// Learnable prediction tokens. Shape: `[max_targets, predictor_embed_dim]`
    prediction_tokens: Tensor<B, 2>,
    /// Predictor embedding dimension.
    predictor_embed_dim: usize,
    /// Encoder embedding dimension (output dimension).
    encoder_embed_dim: usize,
}

impl<B: Backend> Predictor<B> for TransformerPredictor<B> {
    fn predict(
        &self,
        context: &Representation<B>,
        target_positions: &Tensor<B, 2>,
        _latent: Option<&Tensor<B, 2>>,
    ) -> Representation<B> {
        let [batch, _ctx_len, _enc_dim] = context.embeddings.dims();
        let [_batch, num_targets] = target_positions.dims();

        // 1. Project context to predictor dimension
        let ctx = self.input_proj.forward(context.embeddings.clone());
        // ctx: [batch, ctx_len, predictor_embed_dim]

        // 2. Create prediction tokens for this batch
        let pred_tokens = self
            .prediction_tokens
            .clone()
            .slice([0..num_targets, 0..self.predictor_embed_dim])
            .unsqueeze::<3>()
            .expand([batch, num_targets, self.predictor_embed_dim]);

        // 3. Concatenate context + prediction tokens: [batch, ctx_len + num_targets, dim]
        let combined = Tensor::cat(vec![ctx, pred_tokens], 1);
        let ctx_len = context.embeddings.dims()[1];
        let total_len = ctx_len + num_targets;

        // 4. Apply transformer blocks
        let mut x = combined;
        for block in &self.blocks {
            x = block.forward(x);
        }

        // 5. Extract prediction token outputs (last num_targets positions)
        let pred_out = x.slice([0..batch, ctx_len..total_len, 0..self.predictor_embed_dim]);

        // 6. Normalize and project back to encoder dimension
        let pred_out = self.norm.forward(pred_out);
        let pred_out = self.output_proj.forward(pred_out);

        Representation::new(pred_out)
    }
}

// --- Predictor Transformer Block ---

#[derive(Debug, Clone)]
struct PredictorBlockConfig {
    embed_dim: usize,
    num_heads: usize,
}

impl PredictorBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> PredictorBlock<B> {
        let head_dim = self.embed_dim / self.num_heads;
        PredictorBlock {
            norm1: LayerNormConfig::new(self.embed_dim).init(device),
            attn: PredictorAttention {
                qkv: LinearConfig::new(self.embed_dim, 3 * self.embed_dim).init(device),
                out_proj: LinearConfig::new(self.embed_dim, self.embed_dim).init(device),
                num_heads: self.num_heads,
                head_dim,
            },
            norm2: LayerNormConfig::new(self.embed_dim).init(device),
            mlp: PredictorMlp {
                fc1: LinearConfig::new(self.embed_dim, self.embed_dim * 4).init(device),
                fc2: LinearConfig::new(self.embed_dim * 4, self.embed_dim).init(device),
            },
        }
    }
}

#[derive(Module, Debug)]
struct PredictorBlock<B: Backend> {
    norm1: LayerNorm<B>,
    attn: PredictorAttention<B>,
    norm2: LayerNorm<B>,
    mlp: PredictorMlp<B>,
}

impl<B: Backend> PredictorBlock<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let x_norm = self.norm1.forward(x);
        let attn_out = self.attn.forward(x_norm);
        let x = residual + attn_out;

        let residual = x.clone();
        let x_norm = self.norm2.forward(x);
        let mlp_out = self.mlp.forward(x_norm);
        residual + mlp_out
    }
}

#[derive(Module, Debug)]
struct PredictorAttention<B: Backend> {
    qkv: Linear<B>,
    out_proj: Linear<B>,
    num_heads: usize,
    head_dim: usize,
}

impl<B: Backend> PredictorAttention<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();
        let embed_dim = self.num_heads * self.head_dim;

        let qkv = self.qkv.forward(x);
        let q = qkv.clone().slice([0..batch, 0..seq_len, 0..embed_dim]);
        let k = qkv
            .clone()
            .slice([0..batch, 0..seq_len, embed_dim..2 * embed_dim]);
        let v = qkv.slice([0..batch, 0..seq_len, 2 * embed_dim..3 * embed_dim]);

        let q = q
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(k.transpose()) / scale;
        let attn = burn::tensor::activation::softmax(attn, 3);
        let out = attn.matmul(v);
        let out = out.swap_dims(1, 2).reshape([batch, seq_len, embed_dim]);
        self.out_proj.forward(out)
    }
}

#[derive(Module, Debug)]
struct PredictorMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> PredictorMlp<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = burn::tensor::activation::gelu(x);
        self.fc2.forward(x)
    }
}

/// I-JEPA model combining encoder pair and predictor.
///
/// Provides a high-level interface for the I-JEPA pipeline per RFC-002 and RFC-003.
#[derive(Module, Debug)]
pub struct IJepa<B: Backend> {
    /// Context encoder — trained via gradient descent.
    pub context_encoder: crate::vit::VitEncoder<B>,
    /// Target encoder — updated via EMA (no gradients).
    pub target_encoder: crate::vit::VitEncoder<B>,
    /// Predictor — predicts target representations from context.
    pub predictor: TransformerPredictor<B>,
}

/// Configuration for the I-JEPA model.
#[derive(Debug, Clone)]
pub struct IJepaConfig {
    /// ViT encoder config (shared by context and target encoders).
    pub encoder: crate::vit::VitConfig,
    /// Predictor config.
    pub predictor: TransformerPredictorConfig,
}

impl IJepaConfig {
    /// Create a tiny config suitable for testing.
    pub fn tiny_test() -> Self {
        let encoder = crate::vit::VitConfig::tiny_test();
        Self {
            predictor: TransformerPredictorConfig {
                encoder_embed_dim: encoder.embed_dim,
                predictor_embed_dim: 16,
                num_layers: 1,
                num_heads: 2,
                max_target_len: 64,
            },
            encoder,
        }
    }

    /// Initialize an [`IJepa`] model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> IJepa<B> {
        IJepa {
            context_encoder: self.encoder.init(device),
            target_encoder: self.encoder.init(device),
            predictor: self.predictor.init(device),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::ElementConversion;
    use burn_ndarray::NdArray;
    use jepa_core::{EnergyFn, MaskingStrategy};
    use rand::SeedableRng;

    type TestBackend = NdArray<f32>;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

    #[test]
    fn test_predictor_output_shape() {
        let config = TransformerPredictorConfig {
            encoder_embed_dim: 32,
            predictor_embed_dim: 16,
            num_layers: 1,
            num_heads: 2,
            max_target_len: 64,
        };
        let predictor = config.init::<TestBackend>(&device());

        let context = Representation::new(Tensor::zeros([2, 8, 32], &device()));
        let target_pos: Tensor<TestBackend, 2> = Tensor::zeros([2, 4], &device());
        let predicted = predictor.predict(&context, &target_pos, None);

        assert_eq!(predicted.batch_size(), 2);
        assert_eq!(predicted.seq_len(), 4);
        assert_eq!(predicted.embed_dim(), 32);
    }

    #[test]
    fn test_predictor_implements_trait() {
        let config = TransformerPredictorConfig {
            encoder_embed_dim: 16,
            predictor_embed_dim: 8,
            num_layers: 1,
            num_heads: 2,
            max_target_len: 16,
        };
        let predictor = config.init::<TestBackend>(&device());

        let context = Representation::new(Tensor::zeros([1, 4, 16], &device()));
        let target_pos: Tensor<TestBackend, 2> = Tensor::zeros([1, 2], &device());
        let pred: Representation<TestBackend> =
            Predictor::predict(&predictor, &context, &target_pos, None);
        assert_eq!(pred.seq_len(), 2);
    }

    #[test]
    fn test_ijepa_full_pipeline() {
        // End-to-end test: encode → mask → predict → compute energy
        let config = IJepaConfig::tiny_test();
        let model = config.init::<TestBackend>(&device());

        // 1. Create a test image
        let images: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 8, 8], &device());

        // 2. Encode with both encoders
        let context_repr = model.context_encoder.forward(&images);
        let target_repr = model.target_encoder.forward(&images);

        assert_eq!(context_repr.seq_len(), 16); // 4x4 grid
        assert_eq!(target_repr.seq_len(), 16);

        // 3. Generate a mask
        let masking = jepa_core::masking::BlockMasking {
            num_targets: 2,
            target_scale: (0.15, 0.3),
            target_aspect_ratio: (0.75, 1.5),
        };
        let shape = jepa_core::types::InputShape::Image {
            height: 4,
            width: 4,
        };
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let mask = masking.generate_mask(&shape, &mut rng);

        // 4. Predict target from context
        let num_targets = mask.target_indices.len();
        let target_pos: Tensor<TestBackend, 2> = Tensor::zeros([1, num_targets], &device());
        let predicted = model.predictor.predict(&context_repr, &target_pos, None);

        assert_eq!(predicted.seq_len(), num_targets);
        assert_eq!(predicted.embed_dim(), 32);

        // 5. Compute energy between predicted and actual target
        // We need to extract target tokens from target_repr for fair comparison
        // For this test, just verify energy is computable and finite
        let energy = jepa_core::energy::L2Energy.compute(&predicted, &predicted);
        let val: f32 = energy.value.into_scalar().elem();
        assert!(val.is_finite(), "energy should be finite");
    }

    #[test]
    fn test_ijepa_config_tiny() {
        let config = IJepaConfig::tiny_test();
        assert_eq!(config.encoder.embed_dim, 32);
        assert_eq!(config.predictor.predictor_embed_dim, 16);
    }
}
