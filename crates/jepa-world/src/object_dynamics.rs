//! Object-level dynamics predictor for C-JEPA world models.
//!
//! Provides [`ObjectDynamicsPredictor`] — a transformer-based dynamics
//! model that predicts the next state of object-centric representations
//! given the current state and an action.
//!
//! ```text
//! [object slots]  ──► input_proj ──┐
//!                                  ├── concat ──► Transformer ──► output_proj ──► [next slots]
//! [action]        ──► action_proj ─┘
//! ```
//!
//! This module implements [`ActionConditionedPredictor`] and can be used
//! directly with [`WorldModel`](crate::WorldModel) and
//! [`RandomShootingPlanner`](crate::RandomShootingPlanner) for CEM-based
//! planning in object-representation space.
//!
//! The ~98% token reduction (7 object slots vs ~196 patch tokens) makes
//! planning dramatically cheaper than patch-based world models.
//!
//! Reference: Nam et al. (2025), *Causal-JEPA*, §4.3 — CEM planning.

use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::backend::Backend;

use jepa_core::types::Representation;

use crate::action::{Action, ActionConditionedPredictor};

// ---------------------------------------------------------------------------
// Transformer block (self-contained for jepa-world)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DynamicsBlockConfig {
    embed_dim: usize,
    num_heads: usize,
    mlp_dim: usize,
}

impl DynamicsBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> DynamicsBlock<B> {
        let head_dim = self.embed_dim / self.num_heads;
        DynamicsBlock {
            norm1: LayerNormConfig::new(self.embed_dim).init(device),
            qkv: LinearConfig::new(self.embed_dim, 3 * self.embed_dim).init(device),
            out_proj: LinearConfig::new(self.embed_dim, self.embed_dim).init(device),
            norm2: LayerNormConfig::new(self.embed_dim).init(device),
            fc1: LinearConfig::new(self.embed_dim, self.mlp_dim).init(device),
            fc2: LinearConfig::new(self.mlp_dim, self.embed_dim).init(device),
            num_heads: self.num_heads,
            head_dim,
        }
    }
}

/// Pre-norm transformer block for dynamics prediction.
#[derive(Module, Debug)]
struct DynamicsBlock<B: Backend> {
    norm1: LayerNorm<B>,
    qkv: Linear<B>,
    out_proj: Linear<B>,
    norm2: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    num_heads: usize,
    head_dim: usize,
}

impl<B: Backend> DynamicsBlock<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = x.dims();
        let embed_dim = self.num_heads * self.head_dim;

        // Pre-norm self-attention
        let residual = x.clone();
        let x_norm = self.norm1.forward(x);
        let qkv = self.qkv.forward(x_norm);

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
        let attn = burn::tensor::activation::softmax(q.matmul(k.transpose()) / scale, 3);
        let out = attn
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, seq_len, embed_dim]);
        let x = residual + self.out_proj.forward(out);

        // Pre-norm MLP
        let residual = x.clone();
        let mlp_out = self.fc2.forward(burn::tensor::activation::gelu(
            self.fc1.forward(self.norm2.forward(x)),
        ));
        residual + mlp_out
    }
}

// ---------------------------------------------------------------------------
// ObjectDynamicsPredictor
// ---------------------------------------------------------------------------

/// Configuration for [`ObjectDynamicsPredictor`].
///
/// # Example
///
/// ```
/// use jepa_world::object_dynamics::ObjectDynamicsPredictorConfig;
/// use burn_ndarray::NdArray;
///
/// let config = ObjectDynamicsPredictorConfig::tiny_test();
/// let predictor = config.init::<NdArray<f32>>(&burn_ndarray::NdArrayDevice::Cpu);
/// ```
#[derive(Debug, Clone)]
pub struct ObjectDynamicsPredictorConfig {
    /// Object slot dimension (input and output).
    pub slot_dim: usize,
    /// Action dimension.
    pub action_dim: usize,
    /// Internal transformer embedding dimension.
    pub embed_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// MLP hidden dimension.
    pub mlp_dim: usize,
}

impl ObjectDynamicsPredictorConfig {
    /// C-JEPA reference config.
    pub fn cjepa_reference() -> Self {
        Self {
            slot_dim: 128,
            action_dim: 4,
            embed_dim: 1024,
            num_layers: 6,
            num_heads: 16,
            mlp_dim: 2048,
        }
    }

    /// Minimal config for unit tests.
    pub fn tiny_test() -> Self {
        Self {
            slot_dim: 16,
            action_dim: 4,
            embed_dim: 32,
            num_layers: 2,
            num_heads: 4,
            mlp_dim: 64,
        }
    }

    /// Initialize an [`ObjectDynamicsPredictor`] module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ObjectDynamicsPredictor<B> {
        let slot_proj = LinearConfig::new(self.slot_dim, self.embed_dim).init(device);
        let action_proj = LinearConfig::new(self.action_dim, self.embed_dim).init(device);

        let blocks: Vec<DynamicsBlock<B>> = (0..self.num_layers)
            .map(|_| {
                DynamicsBlockConfig {
                    embed_dim: self.embed_dim,
                    num_heads: self.num_heads,
                    mlp_dim: self.mlp_dim,
                }
                .init(device)
            })
            .collect();

        let norm = LayerNormConfig::new(self.embed_dim).init(device);
        let output_proj = LinearConfig::new(self.embed_dim, self.slot_dim).init(device);

        ObjectDynamicsPredictor {
            slot_proj,
            action_proj,
            blocks,
            norm,
            output_proj,
            slot_dim: self.slot_dim,
        }
    }
}

/// Transformer-based dynamics predictor for object-centric world models.
///
/// Given object slots representing the current state and an action,
/// predicts the next state's object slots. The action is projected to
/// the transformer embedding dimension and concatenated as an
/// additional token in the input sequence.
///
/// ```text
/// slot_tokens = SlotProj(object_slots)  [B, N_slots, D]
/// action_token = ActionProj(action)     [B, 1, D]
/// x = [slot_tokens; action_token]       [B, N_slots+1, D]
/// x = Transformer(x)
/// next_slots = OutputProj(x[:, :N_slots, :])  [B, N_slots, slot_dim]
/// ```
#[derive(Module, Debug)]
pub struct ObjectDynamicsPredictor<B: Backend> {
    /// Project object slots to transformer dimension.
    slot_proj: Linear<B>,
    /// Project action to transformer dimension.
    action_proj: Linear<B>,
    /// Transformer layers.
    blocks: Vec<DynamicsBlock<B>>,
    /// Final layer norm.
    norm: LayerNorm<B>,
    /// Project back to slot dimension.
    output_proj: Linear<B>,
    /// Slot dimension (for output extraction).
    slot_dim: usize,
}

impl<B: Backend> ObjectDynamicsPredictor<B> {
    /// Forward pass: predict next object slots from current state and action.
    ///
    /// # Arguments
    /// * `current_slots` - Current object slots. Shape: `[B, N_slots, slot_dim]`
    /// * `action` - Action tensor. Shape: `[B, action_dim]`
    ///
    /// # Returns
    /// Predicted next object slots. Shape: `[B, N_slots, slot_dim]`
    pub fn predict_forward(
        &self,
        current_slots: &Tensor<B, 3>,
        action: &Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let [batch, num_slots, _] = current_slots.dims();

        // Project slots and action
        let slot_tokens = self.slot_proj.forward(current_slots.clone()); // [B, N, D]
        let action_token = self
            .action_proj
            .forward(action.clone())
            .unsqueeze_dim::<3>(1); // [B, 1, D]

        // Concatenate: [slots; action]
        let x = Tensor::cat(vec![slot_tokens, action_token], 1); // [B, N+1, D]

        // Run transformer
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x);
        }
        x = self.norm.forward(x);

        // Extract slot tokens (drop action token) and project back
        let embed_dim = x.dims()[2];
        let slot_out = x.slice([0..batch, 0..num_slots, 0..embed_dim]);
        self.output_proj.forward(slot_out)
    }
}

impl<B: Backend> ActionConditionedPredictor<B> for ObjectDynamicsPredictor<B> {
    fn predict_next_state(
        &self,
        current_state: &Representation<B>,
        action: &Action<B>,
    ) -> Representation<B> {
        let next_slots = self.predict_forward(&current_state.embeddings, &action.data);
        Representation::new(next_slots)
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

    #[test]
    fn test_dynamics_block_output_shape() {
        let block = DynamicsBlockConfig {
            embed_dim: 32,
            num_heads: 4,
            mlp_dim: 64,
        }
        .init::<TestBackend>(&device());

        let x: Tensor<TestBackend, 3> = Tensor::zeros([2, 8, 32], &device());
        let out = block.forward(x);
        assert_eq!(out.dims(), [2, 8, 32]);
    }

    #[test]
    fn test_object_dynamics_predictor_output_shape() {
        let config = ObjectDynamicsPredictorConfig::tiny_test();
        let predictor = config.init::<TestBackend>(&device());

        let slots: Tensor<TestBackend, 3> = Tensor::zeros([2, 5, 16], &device());
        let action: Tensor<TestBackend, 2> = Tensor::zeros([2, 4], &device());
        let next = predictor.predict_forward(&slots, &action);

        assert_eq!(next.dims(), [2, 5, 16]); // same shape as input slots
    }

    #[test]
    fn test_object_dynamics_predictor_output_finite() {
        let config = ObjectDynamicsPredictorConfig::tiny_test();
        let predictor = config.init::<TestBackend>(&device());

        let slots: Tensor<TestBackend, 3> = Tensor::random(
            [1, 4, 16],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device(),
        );
        let action: Tensor<TestBackend, 2> = Tensor::ones([1, 4], &device());
        let next = predictor.predict_forward(&slots, &action);

        let total: f32 = next.abs().sum().into_scalar().elem();
        assert!(total.is_finite(), "output should be finite: {total}");
    }

    #[test]
    fn test_object_dynamics_trait_impl() {
        let config = ObjectDynamicsPredictorConfig::tiny_test();
        let predictor = config.init::<TestBackend>(&device());

        let state = Representation::new(Tensor::zeros([1, 4, 16], &device()));
        let action = Action::new(Tensor::ones([1, 4], &device()));

        let next = predictor.predict_next_state(&state, &action);
        assert_eq!(next.batch_size(), 1);
        assert_eq!(next.seq_len(), 4);
        assert_eq!(next.embed_dim(), 16);
    }

    #[test]
    fn test_different_actions_different_next_states() {
        let config = ObjectDynamicsPredictorConfig::tiny_test();
        let predictor = config.init::<TestBackend>(&device());

        let state = Representation::new(Tensor::zeros([1, 4, 16], &device()));
        let action_a = Action::new(Tensor::zeros([1, 4], &device()));
        let action_b = Action::new(Tensor::ones([1, 4], &device()));

        let next_a = predictor.predict_next_state(&state, &action_a);
        let next_b = predictor.predict_next_state(&state, &action_b);

        let diff: f32 = (next_a.embeddings - next_b.embeddings)
            .abs()
            .sum()
            .into_scalar()
            .elem();
        assert!(
            diff > 1e-6,
            "different actions should produce different states: diff={diff}"
        );
    }

    #[test]
    fn test_predictor_deterministic() {
        let config = ObjectDynamicsPredictorConfig::tiny_test();
        let predictor = config.init::<TestBackend>(&device());

        let state = Representation::new(Tensor::ones([1, 3, 16], &device()));
        let action = Action::new(Tensor::ones([1, 4], &device()));

        let next1 = predictor.predict_next_state(&state, &action);
        let next2 = predictor.predict_next_state(&state, &action);

        let diff: f32 = (next1.embeddings - next2.embeddings)
            .abs()
            .sum()
            .into_scalar()
            .elem();
        assert!(
            diff < 1e-6,
            "predictor should be deterministic: diff={diff}"
        );
    }

    #[test]
    fn test_predictor_with_world_model_rollout() {
        use crate::planner::{L2Cost, WorldModel};

        let config = ObjectDynamicsPredictorConfig::tiny_test();
        let predictor = config.init::<TestBackend>(&device());
        let model = WorldModel::new(predictor, L2Cost);

        let initial = Representation::new(Tensor::zeros([1, 4, 16], &device()));
        let actions = vec![
            Action::new(Tensor::ones([1, 4], &device())),
            Action::new(Tensor::ones([1, 4], &device())),
        ];

        let trajectory = model.rollout(&initial, &actions);
        assert_eq!(trajectory.len(), 3); // initial + 2 predicted

        // States should change with actions
        let diff: f32 = (trajectory[0].embeddings.clone() - trajectory[1].embeddings.clone())
            .abs()
            .sum()
            .into_scalar()
            .elem();
        assert!(diff > 1e-6, "actions should change state");
    }

    #[test]
    fn test_predictor_with_cem_planner() {
        use crate::planner::{L2Cost, RandomShootingConfig, RandomShootingPlanner, WorldModel};
        use rand::SeedableRng;

        let config = ObjectDynamicsPredictorConfig::tiny_test();
        let predictor = config.init::<TestBackend>(&device());
        let model = WorldModel::new(predictor, L2Cost);

        let initial = Representation::new(Tensor::zeros([1, 4, 16], &device()));
        let goal = Representation::new(Tensor::ones([1, 4, 16], &device()));

        let planner = RandomShootingPlanner::new(RandomShootingConfig {
            num_candidates: 16,
            num_iterations: 3,
            num_elites: 4,
            init_std: 1.0,
        });
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let result = planner.plan(&model, &initial, &goal, 2, 4, &mut rng);
        assert_eq!(result.actions.len(), 2);
        assert!(result.cost.is_finite());
        assert_eq!(result.cost_history.len(), 3);
    }
}
