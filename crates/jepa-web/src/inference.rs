//! Inference API exposed to JavaScript.
//!
//! Provides functions to run a forward pass through the ViT encoder and
//! return embedding statistics and per-patch representation norms for
//! visualization.

use burn::prelude::*;
use burn::tensor::backend::Backend;
use burn::tensor::ElementConversion;
use serde::{Deserialize, Serialize};

use jepa_vision::vit::{VitConfig, VitEncoder};

/// Statistics computed from inference output embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Number of patches in the output.
    pub num_patches: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Mean of all embedding values.
    pub mean: f64,
    /// Standard deviation of all embedding values.
    pub std_dev: f64,
    /// Minimum embedding value.
    pub min: f64,
    /// Maximum embedding value.
    pub max: f64,
    /// Per-patch L2 norms (for heatmap visualization).
    pub patch_norms: Vec<f64>,
    /// Grid height (patches).
    pub grid_height: usize,
    /// Grid width (patches).
    pub grid_width: usize,
    /// Inference latency in milliseconds (filled in by JS layer).
    pub latency_ms: f64,
}

/// Run inference on a single image tensor and return statistics.
///
/// The input should be shape `[1, channels, height, width]` matching the
/// encoder's expected input dimensions.
pub fn run_inference<B: Backend>(
    encoder: &VitEncoder<B>,
    input: &Tensor<B, 4>,
    vit_config: &VitConfig,
) -> InferenceResult {
    let repr = encoder.forward(input);
    let embeddings = repr.embeddings.clone();

    let [_batch, num_patches, embed_dim] = embeddings.dims();

    // Global statistics — convert via ElementConversion::elem().
    let flat = embeddings.clone().reshape([num_patches * embed_dim]);
    let mean: f64 = flat.clone().mean().into_scalar().elem();
    let variance: f64 = flat.clone().var(0).into_scalar().elem();
    let std_dev = variance.sqrt();
    let min: f64 = flat.clone().min().into_scalar().elem();
    let max: f64 = flat.max().into_scalar().elem();

    // Per-patch L2 norms for heatmap.
    // embeddings shape: [1, num_patches, embed_dim]
    // Reshape to [num_patches, embed_dim], compute per-row L2 norms.
    let emb_2d = embeddings.reshape([num_patches, embed_dim]);
    let squared = emb_2d.powf_scalar(2.0);
    let summed = squared.sum_dim(1); // [num_patches, 1] or [num_patches]
    let norms = summed.sqrt();
    let norms_data: Vec<f32> = norms.into_data().to_vec().unwrap();
    let patch_norms: Vec<f64> = norms_data.into_iter().map(f64::from).collect();

    let (patch_h, patch_w) = vit_config.patch_size;
    let grid_height = vit_config.image_height / patch_h;
    let grid_width = vit_config.image_width / patch_w;

    InferenceResult {
        num_patches,
        embed_dim,
        mean,
        std_dev,
        min,
        max,
        patch_norms,
        grid_height,
        grid_width,
        latency_ms: 0.0,
    }
}

/// Generate synthetic demo pattern images for inference visualization.
///
/// Returns named patterns as `(name, tensor)` pairs. Each tensor has shape
/// `[1, channels, height, width]`.
pub fn demo_patterns<B: Backend>(
    vit_config: &VitConfig,
    device: &B::Device,
) -> Vec<(&'static str, Tensor<B, 4>)> {
    let c = vit_config.in_channels;
    let h = vit_config.image_height;
    let w = vit_config.image_width;

    let mut patterns = Vec::new();

    // 1. Gradient pattern (left-to-right ramp).
    let mut gradient_data = vec![0.0f32; c * h * w];
    for ch in 0..c {
        for row in 0..h {
            for col in 0..w {
                gradient_data[ch * h * w + row * w + col] = col as f32 / w as f32;
            }
        }
    }
    let gradient = Tensor::from_floats(
        burn::tensor::TensorData::new(gradient_data, [1, c, h, w]),
        device,
    );
    patterns.push(("gradient", gradient));

    // 2. Checkerboard pattern.
    let mut checker_data = vec![0.0f32; c * h * w];
    let block = 2; // checker block size in pixels
    for ch in 0..c {
        for row in 0..h {
            for col in 0..w {
                let value = if (row / block + col / block) % 2 == 0 {
                    1.0
                } else {
                    0.0
                };
                checker_data[ch * h * w + row * w + col] = value;
            }
        }
    }
    let checker = Tensor::from_floats(
        burn::tensor::TensorData::new(checker_data, [1, c, h, w]),
        device,
    );
    patterns.push(("checkerboard", checker));

    // 3. Concentric rings pattern.
    let mut rings_data = vec![0.0f32; c * h * w];
    let center_y = h as f32 / 2.0;
    let center_x = w as f32 / 2.0;
    for ch in 0..c {
        for row in 0..h {
            for col in 0..w {
                let dy = row as f32 - center_y;
                let dx = col as f32 - center_x;
                let dist = (dy * dy + dx * dx).sqrt();
                let value = ((dist * 0.5).sin() + 1.0) / 2.0;
                rings_data[ch * h * w + row * w + col] = value;
            }
        }
    }
    let rings = Tensor::from_floats(
        burn::tensor::TensorData::new(rings_data, [1, c, h, w]),
        device,
    );
    patterns.push(("rings", rings));

    // 4. Random noise.
    let noise: Tensor<B, 4> = Tensor::random(
        [1, c, h, w],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        device,
    );
    patterns.push(("noise", noise));

    patterns
}
