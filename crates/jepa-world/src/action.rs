//! Action types and action-conditioned prediction for world models.
//!
//! Implements RFC-009 (Action-Conditioned World Model).
//!
//! An action-conditioned predictor takes the current state representation
//! and an action, then predicts the next state representation. This enables
//! model-based planning by simulating future trajectories.

use burn::tensor::{backend::Backend, Tensor};

use jepa_core::types::Representation;

/// An action in the environment.
///
/// Wraps an action tensor with semantic meaning. Actions can be
/// continuous (e.g., robot joint torques) or discrete (encoded as one-hot).
///
/// Shape: `[batch, action_dim]`
#[derive(Debug, Clone)]
pub struct Action<B: Backend> {
    /// The action data tensor. Shape: `[batch, action_dim]`
    pub data: Tensor<B, 2>,
}

impl<B: Backend> Action<B> {
    /// Create a new action from a tensor.
    pub fn new(data: Tensor<B, 2>) -> Self {
        Self { data }
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.data.dims()[0]
    }

    /// Get the action dimension.
    pub fn action_dim(&self) -> usize {
        self.data.dims()[1]
    }
}

/// Trait for action-conditioned state predictors.
///
/// Given the current state representation and an action, predicts
/// the next state representation. This is the dynamics model
/// component of a world model.
pub trait ActionConditionedPredictor<B: Backend> {
    /// Predict the next state given current state and action.
    ///
    /// # Arguments
    /// * `current_state` - Current state representation. Shape: `[batch, seq_len, embed_dim]`
    /// * `action` - Action to take. Shape: `[batch, action_dim]`
    ///
    /// # Returns
    /// Predicted next state. Shape: `[batch, seq_len, embed_dim]`
    fn predict_next_state(
        &self,
        current_state: &Representation<B>,
        action: &Action<B>,
    ) -> Representation<B>;
}

/// Errors from world model operations.
#[derive(Debug, thiserror::Error)]
pub enum WorldModelError {
    #[error("batch size mismatch: state has {state} but action has {action}")]
    BatchMismatch { state: usize, action: usize },
    #[error("empty action sequence")]
    EmptyActions,
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::nn::{Linear, LinearConfig};

    use burn::tensor::ElementConversion;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

    #[test]
    fn test_action_creation() {
        let data: Tensor<TestBackend, 2> = Tensor::zeros([2, 4], &device());
        let action = Action::new(data);
        assert_eq!(action.batch_size(), 2);
        assert_eq!(action.action_dim(), 4);
    }

    /// A simple linear dynamics model for testing.
    struct LinearDynamics<B: Backend> {
        state_proj: Linear<B>,
        action_proj: Linear<B>,
    }

    impl ActionConditionedPredictor<TestBackend> for LinearDynamics<TestBackend> {
        fn predict_next_state(
            &self,
            current_state: &Representation<TestBackend>,
            action: &Action<TestBackend>,
        ) -> Representation<TestBackend> {
            let [batch, seq_len, embed_dim] = current_state.embeddings.dims();
            let _device = current_state.embeddings.device();

            // Project state: [batch, seq_len, embed_dim]
            let s = self.state_proj.forward(current_state.embeddings.clone());

            // Project action and broadcast: [batch, 1, embed_dim] → [batch, seq_len, embed_dim]
            let a = self.action_proj.forward(action.data.clone()); // [batch, embed_dim]
            let a = a
                .unsqueeze::<3>() // [batch, 1, embed_dim]
                .expand([batch, seq_len, embed_dim]);

            Representation::new(s + a)
        }
    }

    #[test]
    fn test_action_conditioned_predictor_trait() {
        let embed_dim = 16;
        let action_dim = 4;

        let dynamics = LinearDynamics::<TestBackend> {
            state_proj: LinearConfig::new(embed_dim, embed_dim).init(&device()),
            action_proj: LinearConfig::new(action_dim, embed_dim).init(&device()),
        };

        let state = Representation::new(Tensor::zeros([1, 8, embed_dim], &device()));
        let action = Action::new(Tensor::ones([1, action_dim], &device()));

        let next_state = dynamics.predict_next_state(&state, &action);
        assert_eq!(next_state.batch_size(), 1);
        assert_eq!(next_state.seq_len(), 8);
        assert_eq!(next_state.embed_dim(), embed_dim);
    }

    #[test]
    fn test_different_actions_different_states() {
        let embed_dim = 16;
        let action_dim = 4;

        let dynamics = LinearDynamics::<TestBackend> {
            state_proj: LinearConfig::new(embed_dim, embed_dim).init(&device()),
            action_proj: LinearConfig::new(action_dim, embed_dim).init(&device()),
        };

        let state = Representation::new(Tensor::zeros([1, 4, embed_dim], &device()));
        let action_a = Action::new(Tensor::zeros([1, action_dim], &device()));
        let action_b = Action::new(Tensor::ones([1, action_dim], &device()));

        let next_a = dynamics.predict_next_state(&state, &action_a);
        let next_b = dynamics.predict_next_state(&state, &action_b);

        let diff: f32 = (next_a.embeddings - next_b.embeddings)
            .abs()
            .sum()
            .into_scalar()
            .elem();
        assert!(
            diff > 1e-6,
            "different actions should produce different next states"
        );
    }
}
