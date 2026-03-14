//! Slot Attention for object-centric representation learning.
//!
//! Implements the Slot Attention mechanism (Locatello et al., 2020) and
//! composes it with a frozen ViT backbone to form the [`SlotEncoder`] used
//! in C-JEPA (Nam et al., 2025).
//!
//! ```text
//! [B, C, H, W] ──► Frozen ViT ──► [B, S, D_enc]
//!                                       │
//!                               optional projection
//!                                       │
//!                                       ▼
//!                               SlotAttention
//!                         (iterative slot refinement)
//!                                       │
//!                                       ▼
//!                               [B, N_slots, D_slot]
//! ```
//!
//! References:
//! - Locatello et al. (2020). *Object-Centric Learning with Slot Attention*. NeurIPS.
//! - Zadaianchuk et al. (2023). *VideoSAUR*. NeurIPS.
//! - Nam et al. (2025). *Causal-JEPA*. arXiv:2602.11389.

use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::Backend;

use jepa_core::types::Representation;
use jepa_core::Encoder;

use crate::vit::{VitConfig, VitEncoder};

// ---------------------------------------------------------------------------
// GRU Cell (manual implementation — burn may not expose nn::GruCell)
// ---------------------------------------------------------------------------

/// Configuration for a GRU cell.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GruCellConfig {
    /// Input dimension.
    pub input_dim: usize,
    /// Hidden state dimension.
    pub hidden_dim: usize,
}

impl GruCellConfig {
    /// Initialize a [`GruCell`] module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> GruCell<B> {
        GruCell {
            input_gates: LinearConfig::new(self.input_dim, 3 * self.hidden_dim).init(device),
            hidden_gates: LinearConfig::new(self.hidden_dim, 3 * self.hidden_dim).init(device),
            hidden_dim: self.hidden_dim,
        }
    }
}

/// GRU cell with manual gate equations.
///
/// ```text
/// z = σ(W_iz · x + W_hz · h)          # update gate
/// r = σ(W_ir · x + W_hr · h)          # reset gate
/// n = tanh(W_in · x + r ⊙ (W_hn · h)) # candidate
/// h' = (1 − z) ⊙ n + z ⊙ h           # output
/// ```
#[derive(Module, Debug)]
pub struct GruCell<B: Backend> {
    /// Combined input projections for update, reset, and new gates.
    input_gates: Linear<B>,
    /// Combined hidden projections for update, reset, and new gates.
    hidden_gates: Linear<B>,
    /// Hidden state dimension.
    hidden_dim: usize,
}

impl<B: Backend> GruCell<B> {
    /// Forward pass: `input` `[B, N, D_in]`, `hidden` `[B, N, D_h]` → `[B, N, D_h]`.
    pub fn forward(&self, input: Tensor<B, 3>, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, n, _] = input.dims();
        let d = self.hidden_dim;

        let gi = self.input_gates.forward(input);
        let gh = self.hidden_gates.forward(hidden.clone());

        let gi_z = gi.clone().slice([0..batch, 0..n, 0..d]);
        let gi_r = gi.clone().slice([0..batch, 0..n, d..2 * d]);
        let gi_n = gi.slice([0..batch, 0..n, 2 * d..3 * d]);

        let gh_z = gh.clone().slice([0..batch, 0..n, 0..d]);
        let gh_r = gh.clone().slice([0..batch, 0..n, d..2 * d]);
        let gh_n = gh.slice([0..batch, 0..n, 2 * d..3 * d]);

        let z = burn::tensor::activation::sigmoid(gi_z + gh_z);
        let r = burn::tensor::activation::sigmoid(gi_r + gh_r);
        let n_gate = (gi_n + r * gh_n).tanh();

        let ones = Tensor::<B, 3>::ones(z.dims(), &z.device());
        (ones - z.clone()) * n_gate + z * hidden
    }
}

// ---------------------------------------------------------------------------
// Slot Attention
// ---------------------------------------------------------------------------

/// Configuration for the [`SlotAttention`] module.
///
/// # Example
///
/// ```
/// use jepa_vision::slot_attention::SlotAttentionConfig;
/// use burn_ndarray::NdArray;
///
/// type B = NdArray<f32>;
/// let device = burn_ndarray::NdArrayDevice::Cpu;
///
/// let config = SlotAttentionConfig::tiny_test();
/// let module = config.init::<B>(&device);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SlotAttentionConfig {
    /// Input feature dimension (from backbone).
    pub input_dim: usize,
    /// Slot embedding dimension.
    pub slot_dim: usize,
    /// Number of object slots.
    pub num_slots: usize,
    /// Number of iterative refinement steps.
    pub num_iterations: usize,
    /// Hidden dimension in the post-attention MLP.
    pub mlp_hidden_dim: usize,
}

impl SlotAttentionConfig {
    /// Minimal config for unit tests.
    pub fn tiny_test() -> Self {
        Self {
            input_dim: 32,
            slot_dim: 16,
            num_slots: 4,
            num_iterations: 2,
            mlp_hidden_dim: 32,
        }
    }

    /// C-JEPA reference config (VideoSAUR on DINOv2 ViT-S/14).
    pub fn cjepa_reference() -> Self {
        Self {
            input_dim: 384,
            slot_dim: 128,
            num_slots: 7,
            num_iterations: 3,
            mlp_hidden_dim: 256,
        }
    }

    /// Initialize a [`SlotAttention`] module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SlotAttention<B> {
        let k_proj = LinearConfig::new(self.input_dim, self.slot_dim)
            .with_bias(false)
            .init(device);
        let v_proj = LinearConfig::new(self.input_dim, self.slot_dim)
            .with_bias(false)
            .init(device);
        let q_proj = LinearConfig::new(self.slot_dim, self.slot_dim)
            .with_bias(false)
            .init(device);

        let gru = GruCellConfig {
            input_dim: self.slot_dim,
            hidden_dim: self.slot_dim,
        }
        .init(device);

        let input_norm = LayerNormConfig::new(self.input_dim).init(device);
        let slot_norm = LayerNormConfig::new(self.slot_dim).init(device);
        let mlp_norm = LayerNormConfig::new(self.slot_dim).init(device);
        let mlp_fc1 = LinearConfig::new(self.slot_dim, self.mlp_hidden_dim).init(device);
        let mlp_fc2 = LinearConfig::new(self.mlp_hidden_dim, self.slot_dim).init(device);

        // Learnable slot initialization parameters
        let slot_mu = Tensor::random(
            [self.num_slots, self.slot_dim],
            burn::tensor::Distribution::Normal(0.0, 0.02),
            device,
        );
        let slot_log_sigma = Tensor::zeros([self.num_slots, self.slot_dim], device);

        SlotAttention {
            k_proj,
            v_proj,
            q_proj,
            gru,
            input_norm,
            slot_norm,
            mlp_norm,
            mlp_fc1,
            mlp_fc2,
            slot_mu,
            slot_log_sigma,
            num_slots: self.num_slots,
            slot_dim: self.slot_dim,
            iterations: self.num_iterations,
        }
    }
}

/// Slot Attention module (Locatello et al., 2020).
///
/// Iteratively refines a fixed set of slot vectors by competing for
/// input features via softmax-normalised attention. Each slot learns
/// to bind to a distinct object or visual entity.
///
/// ```text
/// for t in 1..=T:
///   attn = softmax(K(inputs) · Q(slots)^T / √d, dim=slots)
///   updates = WeightedMean(attn, V(inputs))
///   slots = GRU(updates, slots) + MLP(LN(slots))
/// ```
#[derive(Module, Debug)]
pub struct SlotAttention<B: Backend> {
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    q_proj: Linear<B>,
    gru: GruCell<B>,
    input_norm: LayerNorm<B>,
    slot_norm: LayerNorm<B>,
    mlp_norm: LayerNorm<B>,
    mlp_fc1: Linear<B>,
    mlp_fc2: Linear<B>,
    /// Learnable slot mean. Shape: `[num_slots, slot_dim]`.
    slot_mu: Tensor<B, 2>,
    /// Learnable slot log-sigma. Shape: `[num_slots, slot_dim]`.
    slot_log_sigma: Tensor<B, 2>,
    num_slots: usize,
    slot_dim: usize,
    iterations: usize,
}

impl<B: Backend> SlotAttention<B> {
    /// Number of object slots.
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Slot embedding dimension.
    pub fn slot_dim(&self) -> usize {
        self.slot_dim
    }

    /// Run slot attention on input features.
    ///
    /// # Arguments
    /// * `features` - Input features from backbone. Shape: `[B, N_inputs, input_dim]`
    ///
    /// # Returns
    /// Object slots. Shape: `[B, num_slots, slot_dim]`
    pub fn forward(&self, features: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, _, _] = features.dims();
        let device = features.device();

        // Normalize inputs
        let inputs = self.input_norm.forward(features);

        // Project inputs to key/value space
        let k = self.k_proj.forward(inputs.clone()); // [B, N_inputs, slot_dim]
        let v = self.v_proj.forward(inputs); // [B, N_inputs, slot_dim]

        // Initialize slots from learnable parameters + stochastic noise
        let mu =
            self.slot_mu
                .clone()
                .unsqueeze::<3>()
                .expand([batch, self.num_slots, self.slot_dim]);
        let sigma = self.slot_log_sigma.clone().exp().unsqueeze::<3>().expand([
            batch,
            self.num_slots,
            self.slot_dim,
        ]);
        let noise = Tensor::random(
            [batch, self.num_slots, self.slot_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mut slots = mu + sigma * noise;

        let scale = (self.slot_dim as f64).sqrt();

        // Iterative refinement
        for _ in 0..self.iterations {
            let slots_prev = slots.clone();
            let slots_normed = self.slot_norm.forward(slots_prev.clone());

            // Query from slots
            let q = self.q_proj.forward(slots_normed); // [B, N_slots, slot_dim]

            // Attention: softmax over slots for each input position
            // attn_logits: [B, N_inputs, N_slots] = K · Q^T / √d
            let attn_logits = k.clone().matmul(q.transpose()) / scale;
            let attn = burn::tensor::activation::softmax(attn_logits, 2); // [B, N_inputs, N_slots]

            // Weighted mean aggregation
            // attn^T · V: [B, N_slots, N_inputs] · [B, N_inputs, slot_dim] = [B, N_slots, slot_dim]
            let attn_t = attn.transpose(); // [B, N_slots, N_inputs]
            let attn_sum = attn_t.clone().sum_dim(2).clamp_min(1e-8); // [B, N_slots, 1]
            let updates = attn_t.matmul(v.clone()) / attn_sum; // [B, N_slots, slot_dim]

            // GRU update
            slots = self.gru.forward(updates, slots_prev);

            // Residual MLP
            let mlp_in = self.mlp_norm.forward(slots.clone());
            let mlp_out = self
                .mlp_fc2
                .forward(burn::tensor::activation::gelu(self.mlp_fc1.forward(mlp_in)));
            slots = slots + mlp_out;
        }

        slots
    }
}

// ---------------------------------------------------------------------------
// Slot Encoder (frozen backbone + slot attention)
// ---------------------------------------------------------------------------

/// Configuration for the [`SlotEncoder`].
///
/// Composes a (potentially frozen) ViT backbone with Slot Attention
/// to produce object-centric representations from images.
///
/// # Example
///
/// ```
/// use jepa_vision::slot_attention::SlotEncoderConfig;
/// use jepa_vision::vit::VitConfig;
/// use jepa_core::Encoder;
/// use burn_ndarray::NdArray;
///
/// type B = NdArray<f32>;
/// let device = burn_ndarray::NdArrayDevice::Cpu;
///
/// let config = SlotEncoderConfig::tiny_test();
/// let encoder = config.init::<B>(&device);
/// assert_eq!(encoder.embed_dim(), 16);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SlotEncoderConfig {
    /// ViT backbone configuration.
    pub backbone: VitConfig,
    /// Slot attention configuration.
    pub slot_attention: SlotAttentionConfig,
}

impl SlotEncoderConfig {
    /// Minimal configuration for unit tests.
    pub fn tiny_test() -> Self {
        let backbone = VitConfig::tiny_test();
        let backbone_dim = backbone.embed_dim;
        Self {
            backbone,
            slot_attention: SlotAttentionConfig {
                input_dim: backbone_dim,
                slot_dim: 16,
                num_slots: 4,
                num_iterations: 2,
                mlp_hidden_dim: 32,
            },
        }
    }

    /// Initialize a [`SlotEncoder`] module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SlotEncoder<B> {
        let backbone = self.backbone.init(device);
        let slot_attention = self.slot_attention.init(device);

        // Optional projection if backbone dim != slot attention input dim
        let input_proj = if self.backbone.embed_dim != self.slot_attention.input_dim {
            Some(
                LinearConfig::new(self.backbone.embed_dim, self.slot_attention.input_dim)
                    .init(device),
            )
        } else {
            None
        };

        SlotEncoder {
            backbone,
            slot_attention,
            input_proj,
            slot_dim: self.slot_attention.slot_dim,
        }
    }
}

/// Slot Encoder: frozen ViT backbone + Slot Attention.
///
/// Encodes images into object-centric slot representations:
///
/// ```text
/// Image [B, C, H, W]
///   → ViT backbone (frozen) → patch features [B, S, D_enc]
///   → (optional projection) → [B, S, D_input]
///   → Slot Attention → object slots [B, N_slots, D_slot]
/// ```
///
/// In C-JEPA, the backbone is frozen (no gradients) and only the
/// slot attention parameters are trained.
#[derive(Module, Debug)]
pub struct SlotEncoder<B: Backend> {
    /// ViT backbone (typically frozen).
    backbone: VitEncoder<B>,
    /// Slot attention module.
    slot_attention: SlotAttention<B>,
    /// Optional projection from backbone dim to slot attention input dim.
    input_proj: Option<Linear<B>>,
    /// Output slot dimension.
    slot_dim: usize,
}

impl<B: Backend> SlotEncoder<B> {
    /// Forward pass: image → object slots.
    ///
    /// # Arguments
    /// * `images` - Input images. Shape: `[B, C, H, W]`
    ///
    /// # Returns
    /// Object-centric slot representations as a [`Representation`].
    /// Shape: `[B, num_slots, slot_dim]`
    pub fn forward(&self, images: &Tensor<B, 4>) -> Representation<B> {
        // Encode patches with backbone
        let patch_features = self.backbone.forward(images);
        let mut features = patch_features.embeddings;

        // Optional projection
        if let Some(proj) = &self.input_proj {
            features = proj.forward(features);
        }

        // Apply slot attention
        let slots = self.slot_attention.forward(features);
        Representation::new(slots)
    }

    /// Get the output slot dimension.
    pub fn slot_dim(&self) -> usize {
        self.slot_dim
    }

    /// Get the number of slots.
    pub fn num_slots(&self) -> usize {
        self.slot_attention.num_slots()
    }
}

impl<B: Backend> Encoder<B> for SlotEncoder<B> {
    type Input = Tensor<B, 4>;

    fn encode(&self, input: &Self::Input) -> Representation<B> {
        self.forward(input)
    }

    fn embed_dim(&self) -> usize {
        self.slot_dim
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::ElementConversion;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

    // --- GruCell tests ---

    #[test]
    fn test_gru_cell_output_shape() {
        let gru = GruCellConfig {
            input_dim: 16,
            hidden_dim: 16,
        }
        .init::<TestBackend>(&device());

        let input: Tensor<TestBackend, 3> = Tensor::zeros([2, 4, 16], &device());
        let hidden: Tensor<TestBackend, 3> = Tensor::zeros([2, 4, 16], &device());
        let output = gru.forward(input, hidden);
        assert_eq!(output.dims(), [2, 4, 16]);
    }

    #[test]
    fn test_gru_cell_output_finite() {
        let gru = GruCellConfig {
            input_dim: 8,
            hidden_dim: 16,
        }
        .init::<TestBackend>(&device());

        let input: Tensor<TestBackend, 3> = Tensor::random(
            [1, 3, 8],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let hidden: Tensor<TestBackend, 3> = Tensor::zeros([1, 3, 16], &device());
        let output = gru.forward(input, hidden);
        let total: f32 = output.abs().sum().into_scalar().elem();
        assert!(total.is_finite(), "GRU output should be finite: {total}");
    }

    #[test]
    fn test_gru_cell_deterministic() {
        let gru = GruCellConfig {
            input_dim: 8,
            hidden_dim: 8,
        }
        .init::<TestBackend>(&device());

        let input: Tensor<TestBackend, 3> = Tensor::ones([1, 2, 8], &device());
        let hidden: Tensor<TestBackend, 3> = Tensor::zeros([1, 2, 8], &device());
        let out1 = gru.forward(input.clone(), hidden.clone());
        let out2 = gru.forward(input, hidden);
        let diff: f32 = (out1 - out2).abs().sum().into_scalar().elem();
        assert!(diff < 1e-6, "GRU should be deterministic: diff={diff}");
    }

    // --- SlotAttention tests ---

    #[test]
    fn test_slot_attention_output_shape() {
        let config = SlotAttentionConfig::tiny_test();
        let module = config.init::<TestBackend>(&device());

        let features: Tensor<TestBackend, 3> = Tensor::random(
            [2, 16, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let slots = module.forward(features);
        assert_eq!(slots.dims(), [2, 4, 16]); // [batch, num_slots, slot_dim]
    }

    #[test]
    fn test_slot_attention_output_finite() {
        let config = SlotAttentionConfig::tiny_test();
        let module = config.init::<TestBackend>(&device());

        let features: Tensor<TestBackend, 3> = Tensor::random(
            [1, 8, 32],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device(),
        );
        let slots = module.forward(features);
        let total: f32 = slots.abs().sum().into_scalar().elem();
        assert!(
            total.is_finite(),
            "slot attention output should be finite: {total}"
        );
    }

    #[test]
    fn test_slot_attention_accessors() {
        let config = SlotAttentionConfig {
            input_dim: 64,
            slot_dim: 32,
            num_slots: 5,
            num_iterations: 3,
            mlp_hidden_dim: 64,
        };
        let module = config.init::<TestBackend>(&device());
        assert_eq!(module.num_slots(), 5);
        assert_eq!(module.slot_dim(), 32);
    }

    #[test]
    fn test_slot_attention_different_inputs_different_outputs() {
        let config = SlotAttentionConfig::tiny_test();
        let module = config.init::<TestBackend>(&device());

        let a: Tensor<TestBackend, 3> = Tensor::zeros([1, 8, 32], &device());
        let b: Tensor<TestBackend, 3> = Tensor::ones([1, 8, 32], &device());

        let slots_a = module.forward(a);
        let slots_b = module.forward(b);

        let diff: f32 = (slots_a - slots_b).abs().sum().into_scalar().elem();
        assert!(
            diff > 1e-6,
            "different inputs should produce different slots: diff={diff}"
        );
    }

    // --- SlotEncoder tests ---

    #[test]
    fn test_slot_encoder_output_shape() {
        let config = SlotEncoderConfig::tiny_test();
        let encoder = config.init::<TestBackend>(&device());

        let images: Tensor<TestBackend, 4> = Tensor::zeros([2, 1, 8, 8], &device());
        let repr = encoder.forward(&images);

        assert_eq!(repr.batch_size(), 2);
        assert_eq!(repr.seq_len(), 4); // num_slots
        assert_eq!(repr.embed_dim(), 16); // slot_dim
    }

    #[test]
    fn test_slot_encoder_trait_impl() {
        let config = SlotEncoderConfig::tiny_test();
        let encoder = config.init::<TestBackend>(&device());

        let images: Tensor<TestBackend, 4> = Tensor::zeros([1, 1, 8, 8], &device());
        let repr = Encoder::encode(&encoder, &images);

        assert_eq!(repr.batch_size(), 1);
        assert_eq!(repr.seq_len(), 4);
        assert_eq!(encoder.embed_dim(), 16);
    }

    #[test]
    fn test_slot_encoder_output_finite() {
        let config = SlotEncoderConfig::tiny_test();
        let encoder = config.init::<TestBackend>(&device());

        let images: Tensor<TestBackend, 4> = Tensor::random(
            [1, 1, 8, 8],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device(),
        );
        let repr = encoder.forward(&images);
        let total: f32 = repr.embeddings.abs().sum().into_scalar().elem();
        assert!(
            total.is_finite(),
            "slot encoder output should be finite: {total}"
        );
    }

    #[test]
    fn test_slot_encoder_accessors() {
        let config = SlotEncoderConfig::tiny_test();
        let encoder = config.init::<TestBackend>(&device());
        assert_eq!(encoder.slot_dim(), 16);
        assert_eq!(encoder.num_slots(), 4);
    }

    #[test]
    fn test_slot_encoder_with_projection() {
        // Backbone dim (32) != slot attention input_dim (16) — triggers projection
        let config = SlotEncoderConfig {
            backbone: VitConfig::tiny_test(), // embed_dim=32
            slot_attention: SlotAttentionConfig {
                input_dim: 16, // Different from backbone embed_dim
                slot_dim: 8,
                num_slots: 3,
                num_iterations: 1,
                mlp_hidden_dim: 16,
            },
        };
        let encoder = config.init::<TestBackend>(&device());
        assert!(encoder.input_proj.is_some());

        let images: Tensor<TestBackend, 4> = Tensor::zeros([1, 1, 8, 8], &device());
        let repr = encoder.forward(&images);
        assert_eq!(repr.seq_len(), 3);
        assert_eq!(repr.embed_dim(), 8);
    }
}
