//! World model planning via rollout and plan evaluation.
//!
//! Implements RFC-009 (Action-Conditioned World Model) — planning component.
//!
//! A world model combines an encoder, dynamics model, and cost function
//! to enable model-based planning. It can simulate trajectories and
//! evaluate action sequences before execution.

use burn::tensor::backend::Backend;

use jepa_core::types::{Energy, Representation};

use crate::action::{Action, ActionConditionedPredictor};

/// Cost function for evaluating trajectories.
///
/// Measures how far a trajectory's final state is from a goal state.
pub trait CostFunction<B: Backend> {
    /// Compute the cost of a trajectory relative to a goal.
    ///
    /// # Arguments
    /// * `trajectory` - Sequence of state representations
    /// * `goal` - Goal state representation
    ///
    /// # Returns
    /// Cost as an energy value (lower = better)
    fn total_cost(&self, trajectory: &[Representation<B>], goal: &Representation<B>) -> Energy<B>;
}

/// L2 cost: distance between final state and goal in representation space.
pub struct L2Cost;

impl<B: Backend> CostFunction<B> for L2Cost {
    fn total_cost(&self, trajectory: &[Representation<B>], goal: &Representation<B>) -> Energy<B> {
        let final_state = trajectory.last().expect("trajectory must not be empty");
        let diff = final_state.embeddings.clone() - goal.embeddings.clone();
        let cost = (diff.clone() * diff).mean();
        Energy {
            value: cost.unsqueeze(),
        }
    }
}

/// World model that can simulate trajectories and evaluate plans.
///
/// Combines a dynamics model and cost function for model-based planning.
/// The encoder is external — state representations are passed in directly.
pub struct WorldModel<B: Backend, D: ActionConditionedPredictor<B>, C: CostFunction<B>> {
    /// Dynamics model: predicts next state given current state and action.
    pub dynamics: D,
    /// Cost function: evaluates how close a trajectory gets to the goal.
    pub cost: C,
    /// Phantom to hold backend type.
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend, D: ActionConditionedPredictor<B>, C: CostFunction<B>> WorldModel<B, D, C> {
    /// Create a new world model.
    pub fn new(dynamics: D, cost: C) -> Self {
        Self {
            dynamics,
            cost,
            _backend: std::marker::PhantomData,
        }
    }

    /// Simulate a sequence of actions starting from an initial state.
    ///
    /// Returns the full trajectory including the initial state.
    ///
    /// # Arguments
    /// * `initial_state` - Starting state representation
    /// * `actions` - Sequence of actions to simulate
    ///
    /// # Returns
    /// Trajectory of `len(actions) + 1` states (initial + predicted)
    pub fn rollout(
        &self,
        initial_state: &Representation<B>,
        actions: &[Action<B>],
    ) -> Vec<Representation<B>> {
        let mut states = Vec::with_capacity(actions.len() + 1);
        states.push(initial_state.clone());

        for action in actions {
            let next = self
                .dynamics
                .predict_next_state(states.last().unwrap(), action);
            states.push(next);
        }

        states
    }

    /// Evaluate a plan by computing its total cost relative to a goal.
    ///
    /// # Arguments
    /// * `initial_state` - Starting state
    /// * `actions` - Sequence of actions (the plan)
    /// * `goal` - Goal state to reach
    ///
    /// # Returns
    /// Cost of the plan (lower = better)
    pub fn evaluate_plan(
        &self,
        initial_state: &Representation<B>,
        actions: &[Action<B>],
        goal: &Representation<B>,
    ) -> Energy<B> {
        let trajectory = self.rollout(initial_state, actions);
        self.cost.total_cost(&trajectory, goal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn::prelude::*;
    use burn::tensor::ElementConversion;
    use burn_ndarray::NdArray;
    use proptest::prelude::*;

    type TestBackend = NdArray<f32>;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

    /// Simple additive dynamics for testing.
    struct AdditiveDynamics;

    impl ActionConditionedPredictor<TestBackend> for AdditiveDynamics {
        fn predict_next_state(
            &self,
            current_state: &Representation<TestBackend>,
            action: &Action<TestBackend>,
        ) -> Representation<TestBackend> {
            let [batch, seq_len, embed_dim] = current_state.embeddings.dims();
            // Simply add the action (broadcast) to the state
            let a = action
                .data
                .clone()
                .slice([0..batch, 0..embed_dim.min(action.action_dim())])
                .unsqueeze::<3>()
                .expand([batch, seq_len, embed_dim]);
            Representation::new(current_state.embeddings.clone() + a)
        }
    }

    #[test]
    fn test_rollout_length() {
        let model = WorldModel::new(AdditiveDynamics, L2Cost);

        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let actions: Vec<Action<TestBackend>> = (0..10)
            .map(|_| Action::new(Tensor::zeros([1, 8], &device())))
            .collect();

        let trajectory = model.rollout(&initial, &actions);
        assert_eq!(trajectory.len(), 11); // initial + 10 predicted
    }

    #[test]
    fn test_rollout_empty_actions() {
        let model = WorldModel::new(AdditiveDynamics, L2Cost);
        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let trajectory = model.rollout(&initial, &[]);
        assert_eq!(trajectory.len(), 1); // just the initial state
    }

    #[test]
    fn test_rollout_states_change() {
        let model = WorldModel::new(AdditiveDynamics, L2Cost);

        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let actions = vec![Action::new(Tensor::ones([1, 8], &device()))];

        let trajectory = model.rollout(&initial, &actions);
        assert_eq!(trajectory.len(), 2);

        // Second state should differ from initial because action was non-zero
        let diff: f32 = (trajectory[0].embeddings.clone() - trajectory[1].embeddings.clone())
            .abs()
            .sum()
            .into_scalar()
            .elem();
        assert!(diff > 1e-6, "action should change state");
    }

    #[test]
    fn test_evaluate_plan_cost() {
        let model = WorldModel::new(AdditiveDynamics, L2Cost);

        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let goal = Representation::new(Tensor::ones([1, 4, 8], &device()));

        // Plan A: move toward goal
        let plan_a = vec![Action::new(Tensor::ones([1, 8], &device()))];
        // Plan B: move away from goal
        let plan_b = vec![Action::new(Tensor::full([1, 8], -1.0, &device()))];

        let cost_a: f32 = model
            .evaluate_plan(&initial, &plan_a, &goal)
            .value
            .into_scalar()
            .elem();
        let cost_b: f32 = model
            .evaluate_plan(&initial, &plan_b, &goal)
            .value
            .into_scalar()
            .elem();

        assert!(
            cost_a < cost_b,
            "plan toward goal should have lower cost: {cost_a} vs {cost_b}"
        );
    }

    #[test]
    fn test_l2_cost_zero_at_goal() {
        let model = WorldModel::new(AdditiveDynamics, L2Cost);

        let state = Representation::new(Tensor::ones([1, 4, 8], &device()));
        let goal = state.clone();

        let cost: f32 = model
            .evaluate_plan(&state, &[], &goal)
            .value
            .into_scalar()
            .elem();
        assert!(cost.abs() < 1e-6, "cost at goal should be ~0, got {cost}");
    }

    proptest! {
        #[test]
        fn prop_rollout_length_equals_actions_plus_one(num_actions in 0usize..20) {
            let model = WorldModel::new(AdditiveDynamics, L2Cost);
            let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
            let actions: Vec<Action<TestBackend>> = (0..num_actions)
                .map(|_| Action::new(Tensor::zeros([1, 8], &device())))
                .collect();

            let trajectory = model.rollout(&initial, &actions);
            prop_assert_eq!(trajectory.len(), num_actions + 1);
        }

        #[test]
        fn prop_l2_cost_is_non_negative(
            num_actions in 1usize..5,
        ) {
            let model = WorldModel::new(AdditiveDynamics, L2Cost);
            let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
            let goal = Representation::new(Tensor::ones([1, 4, 8], &device()));
            let actions: Vec<Action<TestBackend>> = (0..num_actions)
                .map(|_| Action::new(Tensor::ones([1, 8], &device())))
                .collect();

            let cost: f32 = model
                .evaluate_plan(&initial, &actions, &goal)
                .value
                .into_scalar()
                .elem();
            prop_assert!(cost >= 0.0, "cost was negative: {cost}");
            prop_assert!(cost.is_finite(), "cost was not finite");
        }
    }
}
