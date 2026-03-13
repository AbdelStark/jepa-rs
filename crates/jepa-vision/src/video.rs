//! V-JEPA video encoder with 3D tubelets and 3D RoPE.
//!
//! Implements RFC-002 (Encoder Module) for video input.
//!
//! V-JEPA extends I-JEPA to video by replacing 2D patches with 3D tubelets
//! (temporal × spatial × spatial) and using 3D Rotary Position Encoding
//! for spatiotemporal position awareness.
//!
//! Architecture:
//! 1. Tubelet embedding: video → tubelet sequence
//! 2. 3D RoPE: encode temporal + spatial positions
//! 3. Transformer blocks: self-attention + MLP
//! 4. Layer normalization

use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{backend::Backend, Int, TensorData};

use jepa_core::types::{Energy, MaskError, MaskSpec, Representation};
use jepa_core::{CollapseRegularizer, Encoder, EnergyFn};

/// Configuration for a V-JEPA video encoder.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VitVideoConfig {
    /// Number of input channels (e.g., 3 for RGB).
    pub in_channels: usize,
    /// Number of input frames.
    pub num_frames: usize,
    /// Frame height in pixels.
    pub frame_height: usize,
    /// Frame width in pixels.
    pub frame_width: usize,
    /// Tubelet size `(temporal, height, width)`.
    pub tubelet_size: (usize, usize, usize),
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// MLP hidden dimension (typically 4 * embed_dim).
    pub mlp_dim: usize,
}

impl VitVideoConfig {
    /// Grid dimensions `(temporal, height, width)` after tubelet embedding.
    pub fn grid_dims(&self) -> (usize, usize, usize) {
        (
            self.num_frames / self.tubelet_size.0,
            self.frame_height / self.tubelet_size.1,
            self.frame_width / self.tubelet_size.2,
        )
    }

    /// Total number of tubelets.
    pub fn num_tubelets(&self) -> usize {
        let (gt, gh, gw) = self.grid_dims();
        gt * gh * gw
    }

    /// Create a tiny config for testing.
    pub fn tiny_test() -> Self {
        Self {
            in_channels: 1,
            num_frames: 4,
            frame_height: 8,
            frame_width: 8,
            tubelet_size: (2, 2, 2),
            embed_dim: 32,
            num_layers: 2,
            num_heads: 4,
            mlp_dim: 64,
        }
    }

    /// Initialize a [`VitVideoEncoder`] module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> VitVideoEncoder<B> {
        let tubelet_embed_config = TubeletEmbeddingConfig {
            in_channels: self.in_channels,
            tubelet_t: self.tubelet_size.0,
            tubelet_h: self.tubelet_size.1,
            tubelet_w: self.tubelet_size.2,
            embed_dim: self.embed_dim,
        };
        let tubelet_embed = tubelet_embed_config.init(device);

        let (gt, gh, gw) = self.grid_dims();
        let rope = RotaryPositionEncoding3DConfig::new(self.embed_dim, gt, gh, gw).init(device);

        let blocks: Vec<VideoTransformerBlock<B>> = (0..self.num_layers)
            .map(|_| {
                VideoTransformerBlockConfig {
                    embed_dim: self.embed_dim,
                    num_heads: self.num_heads,
                    mlp_dim: self.mlp_dim,
                }
                .init(device)
            })
            .collect();

        let norm = LayerNormConfig::new(self.embed_dim).init(device);

        VitVideoEncoder {
            tubelet_embed,
            positional_encoding: rope,
            blocks,
            norm,
            embed_dim: self.embed_dim,
        }
    }
}

/// Vision Transformer encoder for video.
///
/// Maps video clips to tubelet-level representations via:
/// 1. Tubelet embedding (linear projection of 3D patches)
/// 2. 3D Rotary Position Encoding (temporal + spatial)
/// 3. Stack of transformer blocks
/// 4. Final layer normalization
///
/// Output shape: `[batch, num_tubelets, embed_dim]`
#[derive(Module, Debug)]
pub struct VitVideoEncoder<B: Backend> {
    /// Tubelet embedding: video → tubelet tokens.
    tubelet_embed: TubeletEmbedding<B>,
    /// 3D Rotary Position Encoding for spatiotemporal positions.
    positional_encoding: RotaryPositionEncoding3D<B>,
    /// Stack of transformer blocks.
    blocks: Vec<VideoTransformerBlock<B>>,
    /// Final layer normalization.
    norm: LayerNorm<B>,
    /// Output embedding dimension.
    embed_dim: usize,
}

impl<B: Backend> VitVideoEncoder<B> {
    fn positioned_tubelet_tokens(&self, video: &Tensor<B, 5>) -> Tensor<B, 3> {
        // 1. Tubelet embedding
        let x = self.tubelet_embed.forward(video.clone());

        // 2. Apply 3D RoPE before masking so surviving tubelets keep their
        // original spatiotemporal coordinates.
        self.positional_encoding.forward(x)
    }

    fn encode_positioned_tokens(&self, mut x: Tensor<B, 3>) -> Representation<B> {
        for block in &self.blocks {
            x = block.forward(x);
        }

        x = self.norm.forward(x);

        Representation::new(x)
    }

    /// Forward pass: video → representation.
    ///
    /// # Arguments
    /// * `video` - Input video. Shape: `[batch, channels, frames, height, width]`
    ///
    /// # Returns
    /// Tubelet-level representations. Shape: `[batch, num_tubelets, embed_dim]`
    pub fn forward(&self, video: &Tensor<B, 5>) -> Representation<B> {
        let x = self.positioned_tubelet_tokens(video);
        self.encode_positioned_tokens(x)
    }

    /// Encode only the visible tubelets for strict JEPA context encoding.
    pub fn forward_visible_tokens(
        &self,
        video: &Tensor<B, 5>,
        visible_indices: &[usize],
    ) -> Representation<B> {
        let x = self.positioned_tubelet_tokens(video);
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

impl<B: Backend> Encoder<B> for VitVideoEncoder<B> {
    type Input = Tensor<B, 5>;

    fn encode(&self, input: &Self::Input) -> Representation<B> {
        self.forward(input)
    }

    fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

// --- Tubelet Embedding ---

/// Configuration for tubelet embedding.
#[derive(Debug, Clone)]
struct TubeletEmbeddingConfig {
    in_channels: usize,
    tubelet_t: usize,
    tubelet_h: usize,
    tubelet_w: usize,
    embed_dim: usize,
}

impl TubeletEmbeddingConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> TubeletEmbedding<B> {
        let tubelet_dim = self.in_channels * self.tubelet_t * self.tubelet_h * self.tubelet_w;
        let projection = LinearConfig::new(tubelet_dim, self.embed_dim).init(device);
        TubeletEmbedding {
            projection,
            tubelet_t: self.tubelet_t,
            tubelet_h: self.tubelet_h,
            tubelet_w: self.tubelet_w,
            in_channels: self.in_channels,
        }
    }
}

/// Tubelet embedding for video.
///
/// Splits a video into non-overlapping 3D tubelets (temporal × height × width)
/// and projects each through a linear layer.
///
/// Input shape: `[batch, channels, frames, height, width]`
/// Output shape: `[batch, num_tubelets, embed_dim]`
#[derive(Module, Debug)]
struct TubeletEmbedding<B: Backend> {
    projection: Linear<B>,
    tubelet_t: usize,
    tubelet_h: usize,
    tubelet_w: usize,
    in_channels: usize,
}

impl<B: Backend> TubeletEmbedding<B> {
    /// Convert a video batch to tubelet embeddings.
    ///
    /// # Arguments
    /// * `video` - Input video. Shape: `[batch, channels, frames, height, width]`
    fn forward(&self, video: Tensor<B, 5>) -> Tensor<B, 3> {
        let [batch, _channels, frames, height, width] = video.dims();

        let grid_t = frames / self.tubelet_t;
        let grid_h = height / self.tubelet_h;
        let grid_w = width / self.tubelet_w;
        let num_tubelets = grid_t * grid_h * grid_w;
        let tubelet_dim = self.in_channels * self.tubelet_t * self.tubelet_h * self.tubelet_w;

        // NdArray supports max 6 dims, so we split into two steps:
        // Step 1: Split temporal axis. [B, C, F, H, W] → [B, C, grid_t, tub_t, H, W]
        let x = video.reshape([
            batch,
            self.in_channels,
            grid_t,
            self.tubelet_t,
            height,
            width,
        ]);
        // Permute to [B, grid_t, C, tub_t, H, W] then flatten: [B*grid_t, C*tub_t, H, W]
        let x = x.permute([0, 2, 1, 3, 4, 5]);
        let c_t = self.in_channels * self.tubelet_t;
        let x: Tensor<B, 4> = x.reshape([batch * grid_t, c_t, height, width]);

        // Step 2: Split spatial axes. [B*grid_t, C*tub_t, H, W] → [B*gt, C*tt, gh, th, gw, tw]
        let x = x.reshape([
            batch * grid_t,
            c_t,
            grid_h,
            self.tubelet_h,
            grid_w,
            self.tubelet_w,
        ]);
        // Permute to [B*gt, gh, gw, C*tt, th, tw]
        let x = x.permute([0, 2, 4, 1, 3, 5]);
        // Flatten: [B*gt, gh*gw, C*tt*th*tw] then reshape to [B, gt*gh*gw, tubelet_dim]
        let spatial_tubelets = grid_h * grid_w;
        let x: Tensor<B, 3> = x.reshape([batch * grid_t, spatial_tubelets, tubelet_dim]);
        let x = x.reshape([batch, num_tubelets, tubelet_dim]);

        // Project: [B, num_tubelets, embed_dim]
        self.projection.forward(x)
    }
}

// --- 3D Rotary Position Encoding ---

/// Configuration for 3D Rotary Position Encoding.
#[derive(Debug, Clone)]
pub struct RotaryPositionEncoding3DConfig {
    /// Embedding dimension (must be divisible by 2).
    pub embed_dim: usize,
    /// Maximum temporal grid size.
    pub max_t: usize,
    /// Maximum spatial grid height.
    pub max_h: usize,
    /// Maximum spatial grid width.
    pub max_w: usize,
    /// Base frequency (default: 10000.0).
    pub base_freq: f64,
}

impl RotaryPositionEncoding3DConfig {
    /// Create a new config.
    pub fn new(embed_dim: usize, max_t: usize, max_h: usize, max_w: usize) -> Self {
        Self {
            embed_dim,
            max_t,
            max_h,
            max_w,
            base_freq: 10000.0,
        }
    }

    /// Initialize the 3D position encoding with precomputed sin/cos tables.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RotaryPositionEncoding3D<B> {
        let half_dim = self.embed_dim / 2;
        // Divide half_dim into 3 parts for temporal, height, width
        // If not perfectly divisible, temporal and height get one extra each
        let sixth = half_dim / 3;
        let dim_t = sixth + (half_dim % 3).min(1);
        let dim_h = sixth + if half_dim % 3 >= 2 { 1 } else { 0 };
        let dim_w = sixth;
        debug_assert_eq!(dim_t + dim_h + dim_w, half_dim);

        let max_seq = self.max_t * self.max_h * self.max_w;

        // Compute frequency bands for each axis
        let freqs_t = compute_freqs(dim_t, self.base_freq, half_dim);
        let freqs_h = compute_freqs(dim_h, self.base_freq, half_dim);
        let freqs_w = compute_freqs(dim_w, self.base_freq, half_dim);

        let mut cos_data = vec![0.0f32; max_seq * half_dim];
        let mut sin_data = vec![0.0f32; max_seq * half_dim];

        for t in 0..self.max_t {
            for h in 0..self.max_h {
                for w in 0..self.max_w {
                    let pos = t * self.max_h * self.max_w + h * self.max_w + w;
                    let mut offset = 0;

                    // Temporal frequencies
                    for (i, &freq) in freqs_t.iter().enumerate() {
                        let angle = t as f64 * freq;
                        cos_data[pos * half_dim + offset + i] = angle.cos() as f32;
                        sin_data[pos * half_dim + offset + i] = angle.sin() as f32;
                    }
                    offset += dim_t;

                    // Height frequencies
                    for (i, &freq) in freqs_h.iter().enumerate() {
                        let angle = h as f64 * freq;
                        cos_data[pos * half_dim + offset + i] = angle.cos() as f32;
                        sin_data[pos * half_dim + offset + i] = angle.sin() as f32;
                    }
                    offset += dim_h;

                    // Width frequencies
                    for (i, &freq) in freqs_w.iter().enumerate() {
                        let angle = w as f64 * freq;
                        cos_data[pos * half_dim + offset + i] = angle.cos() as f32;
                        sin_data[pos * half_dim + offset + i] = angle.sin() as f32;
                    }
                }
            }
        }

        let cos_table = Tensor::from_floats(
            burn::tensor::TensorData::new(cos_data, [max_seq, half_dim]),
            device,
        );
        let sin_table = Tensor::from_floats(
            burn::tensor::TensorData::new(sin_data, [max_seq, half_dim]),
            device,
        );

        RotaryPositionEncoding3D {
            cos_table,
            sin_table,
            embed_dim: self.embed_dim,
        }
    }
}

/// Compute frequency bands for one axis of the 3D RoPE.
fn compute_freqs(num_freqs: usize, base_freq: f64, full_half_dim: usize) -> Vec<f64> {
    (0..num_freqs)
        .map(|i| 1.0 / base_freq.powf(2.0 * i as f64 / full_half_dim as f64))
        .collect()
}

/// 3D Rotary Position Encoding for video.
///
/// Extends RoPE to three dimensions (temporal, height, width) by splitting
/// the embedding dimension into three groups and applying separate rotary
/// frequencies for each spatial/temporal axis.
#[derive(Module, Debug)]
pub struct RotaryPositionEncoding3D<B: Backend> {
    /// Precomputed cosine table. Shape: `[max_seq, half_dim]`
    cos_table: Tensor<B, 2>,
    /// Precomputed sine table. Shape: `[max_seq, half_dim]`
    sin_table: Tensor<B, 2>,
    /// Full embedding dimension.
    embed_dim: usize,
}

impl<B: Backend> RotaryPositionEncoding3D<B> {
    /// Apply 3D rotary encoding to a tensor.
    ///
    /// # Arguments
    /// * `x` - Input tensor. Shape: `[batch, seq_len, embed_dim]`
    ///
    /// # Returns
    /// Rotated tensor with 3D position information encoded. Same shape as input.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _dim] = x.dims();
        let half_dim = self.embed_dim / 2;

        let cos = self.cos_table.clone().slice([0..seq_len, 0..half_dim]);
        let sin = self.sin_table.clone().slice([0..seq_len, 0..half_dim]);

        let cos = cos.unsqueeze::<3>().expand([batch, seq_len, half_dim]);
        let sin = sin.unsqueeze::<3>().expand([batch, seq_len, half_dim]);

        let x1 = x.clone().slice([0..batch, 0..seq_len, 0..half_dim]);
        let x2 = x
            .clone()
            .slice([0..batch, 0..seq_len, half_dim..self.embed_dim]);

        let out1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let out2 = x1 * sin + x2 * cos;

        Tensor::cat(vec![out1, out2], 2)
    }
}

// --- Video Transformer Block ---

#[derive(Debug, Clone)]
struct VideoTransformerBlockConfig {
    embed_dim: usize,
    num_heads: usize,
    mlp_dim: usize,
}

impl VideoTransformerBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> VideoTransformerBlock<B> {
        let head_dim = self.embed_dim / self.num_heads;
        VideoTransformerBlock {
            norm1: LayerNormConfig::new(self.embed_dim).init(device),
            attn: VideoSelfAttention {
                qkv: LinearConfig::new(self.embed_dim, 3 * self.embed_dim).init(device),
                out_proj: LinearConfig::new(self.embed_dim, self.embed_dim).init(device),
                num_heads: self.num_heads,
                head_dim,
            },
            norm2: LayerNormConfig::new(self.embed_dim).init(device),
            mlp: VideoMlp {
                fc1: LinearConfig::new(self.embed_dim, self.mlp_dim).init(device),
                fc2: LinearConfig::new(self.mlp_dim, self.embed_dim).init(device),
            },
        }
    }
}

/// Pre-norm transformer block for video encoder.
#[derive(Module, Debug)]
struct VideoTransformerBlock<B: Backend> {
    norm1: LayerNorm<B>,
    attn: VideoSelfAttention<B>,
    norm2: LayerNorm<B>,
    mlp: VideoMlp<B>,
}

impl<B: Backend> VideoTransformerBlock<B> {
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

/// Multi-head self-attention for video transformer.
#[derive(Module, Debug)]
struct VideoSelfAttention<B: Backend> {
    qkv: Linear<B>,
    out_proj: Linear<B>,
    num_heads: usize,
    head_dim: usize,
}

impl<B: Backend> VideoSelfAttention<B> {
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

/// Two-layer MLP with GELU activation for video transformer.
#[derive(Module, Debug)]
struct VideoMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> VideoMlp<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = burn::tensor::activation::gelu(x);
        self.fc2.forward(x)
    }
}

/// V-JEPA model combining video encoder pair and predictor.
///
/// Provides a high-level interface for the V-JEPA video pipeline per RFC-002 and RFC-003.
/// Uses spatiotemporal masking of tubelets for self-supervised learning on video.
#[derive(Module, Debug)]
pub struct VJepa<B: Backend> {
    /// Context encoder — trained via gradient descent.
    pub context_encoder: VitVideoEncoder<B>,
    /// Target encoder — updated via EMA (no gradients).
    pub target_encoder: VitVideoEncoder<B>,
    /// Predictor — predicts target tubelet representations from context.
    pub predictor: crate::image::TransformerPredictor<B>,
}

/// Output of a strict masked V-JEPA forward step.
#[derive(Debug)]
pub struct StrictVJepaForwardOutput<B: Backend> {
    /// Prediction energy (main loss signal). Shape: `[1]`
    pub energy: Energy<B>,
    /// Collapse prevention regularization loss. Shape: `[1]`
    pub regularization: Tensor<B, 1>,
    /// Total loss (energy + weighted regularization). Shape: `[1]`
    pub total_loss: Tensor<B, 1>,
    /// The mask used for this step.
    pub mask: MaskSpec,
    /// Strictly encoded context representation.
    pub context: Representation<B>,
    /// Predicted target representations.
    pub predicted: Representation<B>,
    /// Actual target representations from the target encoder.
    pub target: Representation<B>,
}

/// Errors returned by [`VJepa::try_forward_step_strict`].
#[derive(Debug, thiserror::Error)]
pub enum StrictVJepaError {
    #[error(transparent)]
    InvalidMask(#[from] MaskError),
    #[error(transparent)]
    Predictor(#[from] crate::image::PredictorError),
}

impl<B: Backend> VJepa<B> {
    /// Encode only the visible tubelets before context self-attention runs.
    ///
    /// This method assumes `context_indices` are already valid for the current
    /// tubelet grid. Use [`VJepa::try_forward_step_strict`] when the indices
    /// come from caller-controlled masking data.
    pub fn encode_context_strict(
        &self,
        video: &Tensor<B, 5>,
        context_indices: &[usize],
    ) -> Representation<B> {
        self.context_encoder
            .forward_visible_tokens(video, context_indices)
    }

    /// Execute a strict masked V-JEPA forward step.
    ///
    /// # Panics
    ///
    /// Panics if `mask` is invalid or if the predictor receives target
    /// positions outside its configured capacity. Use
    /// [`VJepa::try_forward_step_strict`] for typed error reporting.
    pub fn forward_step_strict<EF, CR>(
        &self,
        video: &Tensor<B, 5>,
        mask: MaskSpec,
        energy_fn: &EF,
        regularizer: &CR,
        reg_weight: f64,
    ) -> StrictVJepaForwardOutput<B>
    where
        EF: EnergyFn<B>,
        CR: CollapseRegularizer<B>,
    {
        self.try_forward_step_strict(video, mask, energy_fn, regularizer, reg_weight)
            .expect("strict V-JEPA forward step requires a valid mask and predictor positions within capacity")
    }

    /// Execute a strict masked V-JEPA forward step with typed error reporting.
    pub fn try_forward_step_strict<EF, CR>(
        &self,
        video: &Tensor<B, 5>,
        mask: MaskSpec,
        energy_fn: &EF,
        regularizer: &CR,
        reg_weight: f64,
    ) -> Result<StrictVJepaForwardOutput<B>, StrictVJepaError>
    where
        EF: EnergyFn<B>,
        CR: CollapseRegularizer<B>,
    {
        mask.validate()?;

        let context = self.encode_context_strict(video, &mask.context_indices);
        let target_full = self.target_encoder.forward(video);
        let target = target_full.gather(&mask.target_indices);

        let batch = video.dims()[0];
        let target_positions = crate::image::target_positions_tensor::<B>(
            &mask.target_indices,
            batch,
            &video.device(),
        );
        let predicted = self.predictor.try_predict(&context, &target_positions)?;

        let num_targets = target.seq_len();
        let embed_dim = target.embed_dim();
        let pred_flat = predicted
            .embeddings
            .clone()
            .reshape([batch * num_targets, embed_dim]);
        let target_flat = target
            .embeddings
            .clone()
            .reshape([batch * num_targets, embed_dim]);

        let energy = energy_fn.compute(&predicted, &target);
        let regularization = regularizer.loss(&pred_flat, &target_flat);
        let total_loss = energy.value.clone() + regularization.clone() * reg_weight;

        Ok(StrictVJepaForwardOutput {
            energy,
            regularization,
            total_loss,
            mask,
            context,
            predicted,
            target,
        })
    }
}

/// Configuration for the V-JEPA model.
#[derive(Debug, Clone)]
pub struct VJepaConfig {
    /// Video encoder config (shared by context and target encoders).
    pub encoder: VitVideoConfig,
    /// Predictor config.
    pub predictor: crate::image::TransformerPredictorConfig,
}

impl VJepaConfig {
    /// Create a tiny config suitable for testing.
    pub fn tiny_test() -> Self {
        let encoder = VitVideoConfig::tiny_test();
        Self {
            predictor: crate::image::TransformerPredictorConfig {
                encoder_embed_dim: encoder.embed_dim,
                predictor_embed_dim: 16,
                num_layers: 1,
                num_heads: 2,
                max_target_len: 64,
            },
            encoder,
        }
    }

    /// Initialize a [`VJepa`] model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> VJepa<B> {
        VJepa {
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
    use jepa_core::Predictor;

    type TestBackend = NdArray<f32>;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

    fn fixed_video_mask() -> MaskSpec {
        MaskSpec {
            context_indices: (0..16).collect(),
            target_indices: (16..32).collect(),
            total_tokens: 32,
        }
    }

    fn video_with_hidden_tubelet_value(
        mask: &MaskSpec,
        hidden_value: f32,
    ) -> Tensor<TestBackend, 5> {
        let frames = 4usize;
        let height = 8usize;
        let width = 8usize;
        let mut data = vec![1.0f32; frames * height * width];

        for &index in &mask.target_indices {
            let temporal_block = index / 16;
            let spatial_index = index % 16;
            let spatial_row = spatial_index / 4;
            let spatial_col = spatial_index % 4;

            let frame_start = temporal_block * 2;
            let row_start = spatial_row * 2;
            let col_start = spatial_col * 2;

            for frame in frame_start..frame_start + 2 {
                for row in row_start..row_start + 2 {
                    for col in col_start..col_start + 2 {
                        data[(frame * height + row) * width + col] = hidden_value;
                    }
                }
            }
        }

        Tensor::from_floats(
            burn::tensor::TensorData::new(data, [1, 1, frames, height, width]),
            &device(),
        )
    }

    #[test]
    fn test_vit_video_output_shape() {
        let config = VitVideoConfig::tiny_test();
        let encoder = config.init::<TestBackend>(&device());

        // [batch=2, channels=1, frames=4, height=8, width=8]
        let video: Tensor<TestBackend, 5> = Tensor::zeros([2, 1, 4, 8, 8], &device());
        let repr = encoder.forward(&video);

        // grid: (4/2, 8/2, 8/2) = (2, 4, 4) = 32 tubelets
        assert_eq!(repr.batch_size(), 2);
        assert_eq!(repr.seq_len(), 32);
        assert_eq!(repr.embed_dim(), 32);
    }

    #[test]
    fn test_vit_video_encoder_trait() {
        let config = VitVideoConfig::tiny_test();
        let encoder = config.init::<TestBackend>(&device());

        let video: Tensor<TestBackend, 5> = Tensor::zeros([1, 1, 4, 8, 8], &device());
        let repr = Encoder::encode(&encoder, &video);

        assert_eq!(repr.batch_size(), 1);
        assert_eq!(repr.seq_len(), 32);
        assert_eq!(encoder.embed_dim(), 32);
    }

    #[test]
    fn test_vit_video_different_inputs_different_outputs() {
        let config = VitVideoConfig::tiny_test();
        let encoder = config.init::<TestBackend>(&device());

        let a: Tensor<TestBackend, 5> = Tensor::zeros([1, 1, 4, 8, 8], &device());
        let b: Tensor<TestBackend, 5> = Tensor::ones([1, 1, 4, 8, 8], &device());

        let repr_a = encoder.forward(&a);
        let repr_b = encoder.forward(&b);

        let diff: f32 = (repr_a.embeddings - repr_b.embeddings)
            .abs()
            .sum()
            .into_scalar()
            .elem();
        assert!(
            diff > 1e-6,
            "different video inputs should produce different representations"
        );
    }

    #[test]
    fn test_tubelet_embedding_shape() {
        let config = TubeletEmbeddingConfig {
            in_channels: 3,
            tubelet_t: 2,
            tubelet_h: 16,
            tubelet_w: 16,
            embed_dim: 256,
        };
        let embed = config.init::<TestBackend>(&device());

        // 16 frames, 224x224
        let video: Tensor<TestBackend, 5> = Tensor::zeros([1, 3, 16, 224, 224], &device());
        let out = embed.forward(video);

        // grid: (16/2, 224/16, 224/16) = (8, 14, 14) = 1568 tubelets
        assert_eq!(out.dims(), [1, 1568, 256]);
    }

    #[test]
    fn test_rope3d_output_shape() {
        let config = RotaryPositionEncoding3DConfig::new(64, 2, 4, 4);
        let rope = config.init::<TestBackend>(&device());

        let x: Tensor<TestBackend, 3> = Tensor::ones([2, 32, 64], &device());
        let out = rope.forward(x);
        assert_eq!(out.dims(), [2, 32, 64]);
    }

    #[test]
    fn test_rope3d_preserves_norm() {
        let config = RotaryPositionEncoding3DConfig::new(32, 2, 4, 4);
        let rope = config.init::<TestBackend>(&device());

        let x: Tensor<TestBackend, 3> = Tensor::random(
            [1, 32, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );

        let x_norm: f32 = (x.clone() * x.clone()).sum().into_scalar().elem();
        let out = rope.forward(x);
        let out_norm: f32 = (out.clone() * out.clone()).sum().into_scalar().elem();

        let ratio = out_norm / x_norm;
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "3D RoPE should approximately preserve norm, ratio: {ratio}"
        );
    }

    #[test]
    fn test_rope3d_different_positions_give_different_outputs() {
        let config = RotaryPositionEncoding3DConfig::new(16, 2, 2, 2);
        let rope = config.init::<TestBackend>(&device());

        let x: Tensor<TestBackend, 3> = Tensor::ones([1, 8, 16], &device());
        let out = rope.forward(x);

        // Positions 0 and 1 should differ (different temporal/spatial positions)
        let pos0 = out.clone().slice([0..1, 0..1, 0..16]);
        let pos1 = out.clone().slice([0..1, 1..2, 0..16]);

        let diff: f32 = (pos0 - pos1).abs().sum().into_scalar().elem();
        assert!(
            diff > 1e-6,
            "different 3D positions should produce different outputs"
        );
    }

    #[test]
    fn test_video_config_grid_dims() {
        let config = VitVideoConfig {
            in_channels: 3,
            num_frames: 16,
            frame_height: 224,
            frame_width: 224,
            tubelet_size: (2, 16, 16),
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_dim: 3072,
        };
        assert_eq!(config.grid_dims(), (8, 14, 14));
        assert_eq!(config.num_tubelets(), 1568);
    }

    #[test]
    fn test_video_transformer_block_residual() {
        let block = VideoTransformerBlockConfig {
            embed_dim: 16,
            num_heads: 2,
            mlp_dim: 32,
        }
        .init::<TestBackend>(&device());

        let x: Tensor<TestBackend, 3> = Tensor::zeros([1, 8, 16], &device());
        let out = block.forward(x);
        assert_eq!(out.dims(), [1, 8, 16]);
    }

    #[test]
    fn test_video_self_attention_shape() {
        let attn = VideoSelfAttention {
            qkv: LinearConfig::new(16, 48).init::<TestBackend>(&device()),
            out_proj: LinearConfig::new(16, 16).init::<TestBackend>(&device()),
            num_heads: 4,
            head_dim: 4,
        };

        let x: Tensor<TestBackend, 3> = Tensor::zeros([2, 8, 16], &device());
        let out = attn.forward(x);
        assert_eq!(out.dims(), [2, 8, 16]);
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_video_config_num_tubelets(
            grid_t in 1usize..4,
            grid_h in 1usize..4,
            grid_w in 1usize..4,
        ) {
            let tub = 2;
            let config = VitVideoConfig {
                in_channels: 1,
                num_frames: grid_t * tub,
                frame_height: grid_h * tub,
                frame_width: grid_w * tub,
                tubelet_size: (tub, tub, tub),
                embed_dim: 16,
                num_layers: 1,
                num_heads: 2,
                mlp_dim: 32,
            };
            prop_assert_eq!(config.grid_dims(), (grid_t, grid_h, grid_w));
            prop_assert_eq!(config.num_tubelets(), grid_t * grid_h * grid_w);
        }

        #[test]
        fn prop_rope3d_preserves_shape(
            max_t in 1usize..3,
            max_h in 1usize..3,
            max_w in 1usize..3,
        ) {
            let embed_dim = 12; // divisible by 2, and 6/3=2 per axis
            let config = RotaryPositionEncoding3DConfig::new(embed_dim, max_t, max_h, max_w);
            let rope = config.init::<TestBackend>(&device());
            let seq_len = max_t * max_h * max_w;
            let x: Tensor<TestBackend, 3> = Tensor::ones([1, seq_len, embed_dim], &device());
            let out = rope.forward(x);
            prop_assert_eq!(out.dims(), [1, seq_len, embed_dim]);
        }

        #[test]
        fn prop_rope3d_preserves_norm(
            max_t in 1usize..3,
            max_h in 2usize..4,
            max_w in 2usize..4,
        ) {
            let embed_dim = 12;
            let config = RotaryPositionEncoding3DConfig::new(embed_dim, max_t, max_h, max_w);
            let rope = config.init::<TestBackend>(&device());
            let seq_len = max_t * max_h * max_w;
            let x: Tensor<TestBackend, 3> = Tensor::random(
                [1, seq_len, embed_dim],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device(),
            );
            let x_norm: f32 = (x.clone() * x.clone()).sum().into_scalar().elem();
            let out = rope.forward(x);
            let out_norm: f32 = (out.clone() * out.clone()).sum().into_scalar().elem();
            let ratio = out_norm / x_norm;
            prop_assert!((ratio - 1.0).abs() < 0.01, "3D RoPE norm ratio: {}", ratio);
        }
    }

    // ======================================================================
    // V-JEPA model tests
    // ======================================================================

    #[test]
    fn test_vjepa_config_tiny() {
        let config = VJepaConfig::tiny_test();
        assert_eq!(config.encoder.embed_dim, 32);
        assert_eq!(config.predictor.predictor_embed_dim, 16);
        assert_eq!(config.predictor.encoder_embed_dim, 32);
    }

    #[test]
    fn test_vjepa_model_init() {
        let config = VJepaConfig::tiny_test();
        let model = config.init::<TestBackend>(&device());

        assert_eq!(model.context_encoder.embed_dim, 32);
        assert_eq!(model.target_encoder.embed_dim, 32);
    }

    #[test]
    fn test_strict_video_context_encoding_ignores_hidden_tubelets() {
        let config = VJepaConfig::tiny_test();
        let model = config.init::<TestBackend>(&device());
        let mask = fixed_video_mask();

        let hidden_low = video_with_hidden_tubelet_value(&mask, 0.0);
        let hidden_high = video_with_hidden_tubelet_value(&mask, 1_000.0);

        let strict_low = model.encode_context_strict(&hidden_low, &mask.context_indices);
        let strict_high = model.encode_context_strict(&hidden_high, &mask.context_indices);

        let diff: f32 = (strict_low.embeddings - strict_high.embeddings)
            .abs()
            .sum()
            .into_scalar()
            .elem();
        assert!(
            diff < 1e-5,
            "strict masked video context should ignore hidden tubelets, diff={diff}"
        );
    }

    #[test]
    fn test_full_video_encoder_context_slice_leaks_hidden_tubelets() {
        let config = VitVideoConfig::tiny_test();
        let encoder = config.init::<TestBackend>(&device());
        let mask = fixed_video_mask();

        let hidden_low = video_with_hidden_tubelet_value(&mask, 0.0);
        let hidden_high = video_with_hidden_tubelet_value(&mask, 1_000.0);

        let approx_low = encoder.forward(&hidden_low).gather(&mask.context_indices);
        let approx_high = encoder.forward(&hidden_high).gather(&mask.context_indices);

        let diff: f32 = (approx_low.embeddings - approx_high.embeddings)
            .abs()
            .sum()
            .into_scalar()
            .elem();
        assert!(
            diff > 1e-3,
            "post-encoder gather path should leak hidden tubelets, diff={diff}"
        );
    }

    #[test]
    fn test_strict_video_forward_step_runs_end_to_end() {
        let config = VJepaConfig::tiny_test();
        let model = config.init::<TestBackend>(&device());
        let mask = fixed_video_mask();
        let video = video_with_hidden_tubelet_value(&mask, 5.0);
        let energy_fn = jepa_core::energy::L2Energy;
        let regularizer = jepa_core::collapse::VICReg::default();

        let output = model.forward_step_strict(&video, mask.clone(), &energy_fn, &regularizer, 1.0);

        assert_eq!(output.context.seq_len(), mask.context_indices.len());
        assert_eq!(output.predicted.seq_len(), mask.target_indices.len());
        assert_eq!(output.target.seq_len(), mask.target_indices.len());

        let total_loss: f32 = output.total_loss.into_scalar().elem();
        assert!(
            total_loss.is_finite(),
            "strict video forward loss should be finite"
        );
    }

    #[test]
    fn test_try_strict_video_forward_step_rejects_invalid_mask() {
        let config = VJepaConfig::tiny_test();
        let model = config.init::<TestBackend>(&device());
        let video = Tensor::ones([1, 1, 4, 8, 8], &device());
        let invalid_mask = MaskSpec {
            context_indices: vec![0],
            target_indices: vec![],
            total_tokens: 32,
        };
        let energy_fn = jepa_core::energy::L2Energy;
        let regularizer = jepa_core::collapse::VICReg::default();

        let err = model
            .try_forward_step_strict(&video, invalid_mask, &energy_fn, &regularizer, 1.0)
            .unwrap_err();
        assert!(matches!(
            err,
            StrictVJepaError::InvalidMask(MaskError::EmptyTarget)
        ));
    }

    /// BDD: "V-JEPA full pipeline with spatiotemporal masking"
    /// Given a V-JEPA model with video encoder pair and predictor
    /// When I encode a video clip, generate a spatiotemporal mask,
    ///   gather target tubelets, and predict from context
    /// Then the energy should be finite and non-negative
    #[test]
    fn bdd_vjepa_full_pipeline_with_spatiotemporal_masking() {
        use jepa_core::{CollapseRegularizer, EnergyFn, MaskingStrategy};
        use rand::SeedableRng;

        let config = VJepaConfig::tiny_test();
        let model = config.init::<TestBackend>(&device());

        // Video: [batch=1, channels=1, frames=4, height=8, width=8]
        let video: Tensor<TestBackend, 5> = Tensor::random(
            [1, 1, 4, 8, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );

        // 1. Encode with both encoders
        let context_repr = model.context_encoder.forward(&video);
        let target_repr = model.target_encoder.forward(&video);

        // grid: (4/2, 8/2, 8/2) = (2, 4, 4) = 32 tubelets
        assert_eq!(context_repr.seq_len(), 32);
        assert_eq!(target_repr.seq_len(), 32);

        // 2. Generate spatiotemporal mask
        let masking = jepa_core::masking::SpatiotemporalMasking {
            num_targets: 2,
            temporal_extent: (1, 2),
            spatial_scale: (0.1, 0.2),
        };
        let shape = jepa_core::types::InputShape::Video {
            frames: 2,
            height: 4,
            width: 4,
        };
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let mask = masking.generate_mask(&shape, &mut rng);
        assert!(mask.validate().is_ok());
        assert_eq!(mask.context_indices.len() + mask.target_indices.len(), 32);

        // 3. Gather target tubelets
        let target_gathered = target_repr.gather(&mask.target_indices);
        assert_eq!(target_gathered.seq_len(), mask.target_indices.len());

        // 4. Predict targets from context
        let num_targets = mask.target_indices.len();
        let target_pos: Tensor<TestBackend, 2> = Tensor::zeros([1, num_targets], &device());
        let predicted = model.predictor.predict(&context_repr, &target_pos, None);
        assert_eq!(predicted.seq_len(), num_targets);
        assert_eq!(predicted.embed_dim(), 32);

        // 5. Compute energy
        let energy = jepa_core::energy::L2Energy.compute(&predicted, &target_gathered);
        let val: f32 = energy.value.into_scalar().elem();
        assert!(val.is_finite(), "energy should be finite, got {val}");
        assert!(val >= 0.0, "L2 energy should be non-negative, got {val}");

        // 6. Collapse regularization
        let embed_dim = predicted.embed_dim();
        let pred_flat = predicted.embeddings.reshape([num_targets, embed_dim]);
        let target_flat = target_gathered.embeddings.reshape([num_targets, embed_dim]);
        let reg: f32 = jepa_core::collapse::VICReg::default()
            .loss(&pred_flat, &target_flat)
            .into_scalar()
            .elem();
        assert!(
            reg.is_finite(),
            "regularization should be finite, got {reg}"
        );
    }
}
