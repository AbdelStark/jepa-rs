//! Rotary Position Embedding (RoPE) for 2D spatial positions.
//!
//! Implements position encoding from RFC-002 (Encoder Module).
//!
//! RoPE encodes absolute position information by rotating the query and key
//! vectors in attention. For 2D images, we use separate rotary frequencies
//! for the height and width axes, concatenating them along the embedding dim.
//!
//! Reference: RoFormer (Su et al., 2021) extended to 2D for vision.

use burn::prelude::*;
use burn::tensor::backend::Backend;

/// Configuration for 2D Rotary Position Embedding.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RotaryPositionEncoding2DConfig {
    /// Embedding dimension (must be even for rotation pairs).
    pub embed_dim: usize,
    /// Maximum grid height (number of patch rows).
    pub max_height: usize,
    /// Maximum grid width (number of patch columns).
    pub max_width: usize,
    /// Base frequency for the sinusoidal encoding (default: 10000.0).
    pub base_freq: f64,
}

impl RotaryPositionEncoding2DConfig {
    /// Create a new config.
    pub fn new(embed_dim: usize, max_height: usize, max_width: usize) -> Self {
        Self {
            embed_dim,
            max_height,
            max_width,
            base_freq: 10000.0,
        }
    }

    /// Initialize the position encoding, precomputing sin/cos tables.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RotaryPositionEncoding2D<B> {
        let half_dim = self.embed_dim / 2;
        let quarter_dim = half_dim / 2;
        let max_seq = self.max_height * self.max_width;

        // Compute frequency bands: freq_i = 1 / (base ^ (2i / dim))
        let mut freqs_data = Vec::with_capacity(quarter_dim);
        for i in 0..quarter_dim {
            let freq = 1.0 / self.base_freq.powf(2.0 * i as f64 / half_dim as f64);
            freqs_data.push(freq as f32);
        }

        // Build position-frequency tables for height and width
        let mut cos_data = vec![0.0f32; max_seq * half_dim];
        let mut sin_data = vec![0.0f32; max_seq * half_dim];

        for row in 0..self.max_height {
            for col in 0..self.max_width {
                let pos = row * self.max_width + col;
                // First quarter_dim: height frequencies
                for (i, &freq) in freqs_data.iter().enumerate() {
                    let angle = row as f64 * freq as f64;
                    cos_data[pos * half_dim + i] = angle.cos() as f32;
                    sin_data[pos * half_dim + i] = angle.sin() as f32;
                }
                // Second quarter_dim: width frequencies
                for (i, &freq) in freqs_data.iter().enumerate() {
                    let angle = col as f64 * freq as f64;
                    cos_data[pos * half_dim + quarter_dim + i] = angle.cos() as f32;
                    sin_data[pos * half_dim + quarter_dim + i] = angle.sin() as f32;
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

        RotaryPositionEncoding2D {
            cos_table,
            sin_table,
            embed_dim: self.embed_dim,
        }
    }
}

/// 2D Rotary Position Embedding.
///
/// Applies rotary encoding to query/key tensors by rotating pairs of
/// dimensions using precomputed sin/cos tables derived from 2D grid positions.
#[derive(Module, Debug)]
pub struct RotaryPositionEncoding2D<B: Backend> {
    /// Precomputed cosine table. Shape: `[max_seq, half_dim]`
    cos_table: Tensor<B, 2>,
    /// Precomputed sine table. Shape: `[max_seq, half_dim]`
    sin_table: Tensor<B, 2>,
    /// Full embedding dimension.
    embed_dim: usize,
}

impl<B: Backend> RotaryPositionEncoding2D<B> {
    /// Apply rotary encoding to a tensor.
    ///
    /// # Arguments
    /// * `x` - Input tensor. Shape: `[batch, seq_len, embed_dim]`
    ///
    /// # Returns
    /// Rotated tensor with position information encoded. Same shape as input.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _dim] = x.dims();
        let half_dim = self.embed_dim / 2;

        // Slice cos/sin tables to current seq_len
        let cos = self.cos_table.clone().slice([0..seq_len, 0..half_dim]); // [seq_len, half_dim]
        let sin = self.sin_table.clone().slice([0..seq_len, 0..half_dim]); // [seq_len, half_dim]

        // Unsqueeze for broadcasting over batch: [1, seq_len, half_dim]
        let cos = cos.unsqueeze::<3>().expand([batch, seq_len, half_dim]);
        let sin = sin.unsqueeze::<3>().expand([batch, seq_len, half_dim]);

        // Split x into two halves
        let x1 = x.clone().slice([0..batch, 0..seq_len, 0..half_dim]);
        let x2 = x
            .clone()
            .slice([0..batch, 0..seq_len, half_dim..self.embed_dim]);

        // Apply rotation: [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        let out1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let out2 = x1 * sin + x2 * cos;

        Tensor::cat(vec![out1, out2], 2)
    }

    /// Get the embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::ElementConversion;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

    #[test]
    fn test_rope_output_shape() {
        let config = RotaryPositionEncoding2DConfig::new(64, 14, 14);
        let rope = config.init::<TestBackend>(&device());

        let x: Tensor<TestBackend, 3> = Tensor::ones([2, 196, 64], &device());
        let out = rope.forward(x);
        assert_eq!(out.dims(), [2, 196, 64]);
    }

    #[test]
    fn test_rope_preserves_norm_approximately() {
        // RoPE is a rotation, so it should approximately preserve vector norms
        let config = RotaryPositionEncoding2DConfig::new(32, 4, 4);
        let rope = config.init::<TestBackend>(&device());

        let x: Tensor<TestBackend, 3> = Tensor::random(
            [1, 16, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );

        let x_norm: f32 = (x.clone() * x.clone()).sum().into_scalar().elem();

        let out = rope.forward(x);
        let out_norm: f32 = (out.clone() * out.clone()).sum().into_scalar().elem();

        let ratio = out_norm / x_norm;
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "RoPE should approximately preserve norm, ratio: {ratio}"
        );
    }

    #[test]
    fn test_rope_different_positions_give_different_outputs() {
        let config = RotaryPositionEncoding2DConfig::new(16, 4, 4);
        let rope = config.init::<TestBackend>(&device());

        // Same vector at all positions
        let x: Tensor<TestBackend, 3> = Tensor::ones([1, 16, 16], &device());
        let out = rope.forward(x);

        // Extract position 0 and position 1
        let pos0 = out.clone().slice([0..1, 0..1, 0..16]);
        let pos1 = out.clone().slice([0..1, 1..2, 0..16]);

        // They should be different because of position encoding
        let diff: f32 = (pos0 - pos1).abs().sum().into_scalar().elem();
        assert!(
            diff > 1e-6,
            "different positions should produce different outputs"
        );
    }

    #[test]
    fn test_rope_small_grid() {
        let config = RotaryPositionEncoding2DConfig::new(8, 2, 2);
        let rope = config.init::<TestBackend>(&device());

        let x: Tensor<TestBackend, 3> = Tensor::ones([1, 4, 8], &device());
        let out = rope.forward(x);
        assert_eq!(out.dims(), [1, 4, 8]);
    }
}
