//! World model planning via rollout and plan evaluation.
//!
//! Implements RFC-009 (Action-Conditioned World Model) — planning component.
//!
//! A world model combines an encoder, dynamics model, and cost function
//! to enable **model-based planning**: simulate candidate trajectories in
//! representation space, score them, and select the best action sequence.
//!
//! The planner uses the **Cross-Entropy Method (CEM)** — a derivative-free
//! optimizer that iteratively:
//! 1. Samples random action sequences from a Gaussian.
//! 2. Rolls out the dynamics model to get trajectories.
//! 3. Evaluates each trajectory with a [`CostFunction`].
//! 4. Refits the Gaussian to the top-*k* elite sequences.
//!
//! This approach works well even when the dynamics model and cost function
//! are non-differentiable.

use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Tensor};
use rand::RngExt as _;

use jepa_core::types::{Energy, Representation};

use crate::action::{Action, ActionConditionedPredictor};

/// Cost function for evaluating trajectories.
///
/// Measures how far a trajectory's final state is from a goal state.
pub trait CostFunction<B: Backend> {
    /// Compute the cost of a trajectory relative to a goal.
    ///
    /// # Arguments
    /// * `trajectory` - Sequence of state representations (must be non-empty)
    /// * `goal` - Goal state representation
    ///
    /// # Panics
    /// Implementations may panic if `trajectory` is empty.
    ///
    /// # Returns
    /// Cost as an energy value (lower = better)
    fn total_cost(&self, trajectory: &[Representation<B>], goal: &Representation<B>) -> Energy<B>;
}

/// Errors from planning and cost evaluation helpers.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PlanningError {
    #[error("trajectory must not be empty")]
    EmptyTrajectory,
    #[error("num_candidates must be positive, got {0}")]
    ZeroCandidates(usize),
    #[error("num_iterations must be positive, got {0}")]
    ZeroIterations(usize),
    #[error("num_elites must be positive, got {0}")]
    ZeroElites(usize),
    #[error("planning horizon must be positive, got {0}")]
    ZeroHorizon(usize),
    #[error("action_dim must be positive, got {0}")]
    ZeroActionDim(usize),
}

/// L2 cost: distance between final state and goal in representation space.
///
/// # Example
///
/// ```
/// use burn::prelude::*;
/// use burn_ndarray::NdArray;
/// use jepa_core::types::Representation;
/// use jepa_world::planner::{L2Cost, CostFunction};
///
/// type B = NdArray<f32>;
/// let device = burn_ndarray::NdArrayDevice::Cpu;
///
/// let cost = L2Cost;
/// let state: Representation<B> = Representation::new(Tensor::zeros([1, 4, 8], &device));
/// let goal: Representation<B> = Representation::new(Tensor::ones([1, 4, 8], &device));
///
/// let trajectory = vec![state];
/// let energy = cost.total_cost(&trajectory, &goal);
/// // Energy should be positive when state differs from goal
/// let dims = energy.value.dims();
/// assert_eq!(dims, [1]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct L2Cost;

impl L2Cost {
    /// Fallible cost evaluation for caller-controlled trajectories.
    pub fn try_total_cost<B: Backend>(
        &self,
        trajectory: &[Representation<B>],
        goal: &Representation<B>,
    ) -> Result<Energy<B>, PlanningError> {
        let Some(final_state) = trajectory.last() else {
            return Err(PlanningError::EmptyTrajectory);
        };

        let diff = final_state.embeddings.clone() - goal.embeddings.clone();
        let cost = (diff.clone() * diff).mean();
        Ok(Energy {
            value: cost.unsqueeze(),
        })
    }
}

impl<B: Backend> CostFunction<B> for L2Cost {
    /// # Panics
    ///
    /// Panics if `trajectory` is empty. Use [`L2Cost::try_total_cost`] when
    /// the caller controls the trajectory contents.
    fn total_cost(&self, trajectory: &[Representation<B>], goal: &Representation<B>) -> Energy<B> {
        self.try_total_cost(trajectory, goal)
            .expect("CostFunction::total_cost requires a non-empty trajectory; use try_total_cost for error handling")
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
            // Safety: states is non-empty — we pushed initial_state above and push each iteration
            let next = self
                .dynamics
                .predict_next_state(states.last().expect("states is non-empty"), action);
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

/// Configuration for the random-shooting planner (CEM-style).
///
/// # Example
///
/// ```
/// use jepa_world::planner::RandomShootingConfig;
///
/// let config = RandomShootingConfig {
///     num_candidates: 128,
///     num_iterations: 10,
///     num_elites: 16,
///     init_std: 2.0,
/// };
/// assert_eq!(config.num_candidates, 128);
///
/// // Default configuration is also available:
/// let default = RandomShootingConfig::default();
/// assert_eq!(default.num_candidates, 64);
/// ```
#[derive(Debug, Clone)]
pub struct RandomShootingConfig {
    /// Number of candidate action sequences to sample per iteration.
    pub num_candidates: usize,
    /// Number of optimization iterations.
    pub num_iterations: usize,
    /// Number of top candidates to keep (elite set) for refining the distribution.
    pub num_elites: usize,
    /// Initial standard deviation for action sampling.
    pub init_std: f64,
}

impl Default for RandomShootingConfig {
    fn default() -> Self {
        Self {
            num_candidates: 64,
            num_iterations: 5,
            num_elites: 8,
            init_std: 1.0,
        }
    }
}

impl RandomShootingConfig {
    /// Validate the planner configuration.
    pub fn validate(&self) -> Result<(), PlanningError> {
        if self.num_candidates == 0 {
            return Err(PlanningError::ZeroCandidates(self.num_candidates));
        }
        if self.num_iterations == 0 {
            return Err(PlanningError::ZeroIterations(self.num_iterations));
        }
        if self.num_elites == 0 {
            return Err(PlanningError::ZeroElites(self.num_elites));
        }
        Ok(())
    }
}

/// Floor for the per-dimension standard deviation during CEM distribution
/// refit, preventing the search from collapsing to a single point.
const MIN_CEM_STD: f64 = 0.01;

/// Convert a sequence of f64 action values into an [`Action`] tensor.
fn action_from_floats<B: Backend>(
    values: &[f64],
    action_dim: usize,
    device: &B::Device,
) -> Action<B> {
    let data: Vec<f32> = values.iter().map(|&v| v as f32).collect();
    Action::new(Tensor::from_floats(
        burn::tensor::TensorData::new(data, [1, action_dim]),
        device,
    ))
}

/// Random-shooting planner (Cross-Entropy Method).
///
/// Optimizes action sequences by:
/// 1. Sampling candidate action sequences from a Gaussian distribution
/// 2. Evaluating each candidate via world model rollout
/// 3. Selecting the top-k (elite) candidates
/// 4. Refitting the Gaussian to the elite set
/// 5. Repeating for several iterations
///
/// This is a zeroth-order optimization method that works with any backend
/// (no autodiff required).
#[derive(Debug, Clone)]
pub struct RandomShootingPlanner {
    /// Planner configuration.
    pub config: RandomShootingConfig,
}

/// Output of the planning process.
#[derive(Debug)]
pub struct PlanResult<B: Backend> {
    /// The best action sequence found.
    pub actions: Vec<Action<B>>,
    /// The cost of the best plan.
    pub cost: f32,
    /// Cost history: best cost at each iteration.
    pub cost_history: Vec<f32>,
}

impl RandomShootingPlanner {
    /// Create a new random-shooting planner with the given configuration.
    ///
    /// This constructor preserves the historical panic-on-use behavior of
    /// [`RandomShootingPlanner::plan`]. Use [`RandomShootingPlanner::try_new`]
    /// when the configuration comes from untrusted or caller-controlled input.
    pub fn new(config: RandomShootingConfig) -> Self {
        Self { config }
    }

    /// Create a new planner after validating the configuration.
    pub fn try_new(config: RandomShootingConfig) -> Result<Self, PlanningError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Plan an action sequence to reach a goal state.
    ///
    /// Uses the Cross-Entropy Method (CEM) to iteratively refine a distribution
    /// over action sequences, selecting the best trajectory under the world model.
    ///
    /// # Arguments
    /// * `world_model` - The world model for rollout and cost evaluation
    /// * `initial_state` - Starting state
    /// * `goal` - Goal state to reach
    /// * `horizon` - Number of actions in the plan
    /// * `action_dim` - Dimension of each action vector
    /// * `rng` - Random number generator
    ///
    /// # Panics
    ///
    /// Panics if the planner configuration is invalid or if `horizon` /
    /// `action_dim` are zero. Use [`RandomShootingPlanner::try_plan`] for
    /// typed error reporting on caller-controlled inputs.
    pub fn plan<B: Backend, D: ActionConditionedPredictor<B>, C: CostFunction<B>>(
        &self,
        world_model: &WorldModel<B, D, C>,
        initial_state: &Representation<B>,
        goal: &Representation<B>,
        horizon: usize,
        action_dim: usize,
        rng: &mut impl rand::Rng,
    ) -> PlanResult<B> {
        self.try_plan(world_model, initial_state, goal, horizon, action_dim, rng)
            .expect(
                "RandomShootingPlanner::plan failed — horizon, action_dim, candidates, \
                 iterations, and elites must all be > 0; use try_plan for error handling",
            )
    }

    /// Plan an action sequence with typed error reporting for invalid inputs.
    pub fn try_plan<B: Backend, D: ActionConditionedPredictor<B>, C: CostFunction<B>>(
        &self,
        world_model: &WorldModel<B, D, C>,
        initial_state: &Representation<B>,
        goal: &Representation<B>,
        horizon: usize,
        action_dim: usize,
        rng: &mut impl rand::Rng,
    ) -> Result<PlanResult<B>, PlanningError> {
        self.config.validate()?;
        if horizon == 0 {
            return Err(PlanningError::ZeroHorizon(horizon));
        }
        if action_dim == 0 {
            return Err(PlanningError::ZeroActionDim(action_dim));
        }

        let device = initial_state.embeddings.device();

        // Initialize mean and std for the action distribution
        let mut mean = vec![vec![0.0f64; action_dim]; horizon];
        let mut std = vec![vec![self.config.init_std; action_dim]; horizon];

        let mut cost_history = Vec::with_capacity(self.config.num_iterations);
        let mut best_actions: Vec<Action<B>> = Vec::new();
        let mut best_cost = f32::MAX;

        for _iter in 0..self.config.num_iterations {
            // 1. Sample candidates
            let mut candidates: Vec<Vec<Vec<f64>>> = Vec::with_capacity(self.config.num_candidates);
            for _ in 0..self.config.num_candidates {
                let mut candidate = Vec::with_capacity(horizon);
                for t in 0..horizon {
                    let action_vals: Vec<f64> = (0..action_dim)
                        .map(|d| {
                            let noise: f64 = rng.random::<f64>() * 2.0 - 1.0; // uniform [-1, 1]
                            mean[t][d] + std[t][d] * noise
                        })
                        .collect();
                    candidate.push(action_vals);
                }
                candidates.push(candidate);
            }

            // 2. Evaluate each candidate
            let mut costs: Vec<(usize, f32)> = candidates
                .iter()
                .enumerate()
                .map(|(i, candidate)| {
                    let actions: Vec<Action<B>> = candidate
                        .iter()
                        .map(|a| action_from_floats(a, action_dim, &device))
                        .collect();

                    let cost: f32 = world_model
                        .evaluate_plan(initial_state, &actions, goal)
                        .value
                        .into_scalar()
                        .elem();
                    (i, cost)
                })
                .collect();

            // 3. Sort by cost and select elites.
            // Use f32::total_cmp so NaN/Inf values sort deterministically
            // instead of silently corrupting the elite set.
            costs.sort_by(|a, b| a.1.total_cmp(&b.1));
            let num_elites = self.config.num_elites.min(costs.len());

            // Track best (use total_cmp so NaN is handled consistently)
            if costs[0].1.total_cmp(&best_cost).is_lt() {
                best_cost = costs[0].1;
                let best_idx = costs[0].0;
                best_actions = candidates[best_idx]
                    .iter()
                    .map(|a| action_from_floats(a, action_dim, &device))
                    .collect();
            }
            cost_history.push(best_cost);

            // 4. Refit distribution to elites
            let elite_indices: Vec<usize> = costs[..num_elites].iter().map(|(i, _)| *i).collect();

            for t in 0..horizon {
                for d in 0..action_dim {
                    let elite_vals: Vec<f64> =
                        elite_indices.iter().map(|&i| candidates[i][t][d]).collect();
                    let n = elite_vals.len() as f64;
                    let new_mean = elite_vals.iter().sum::<f64>() / n;
                    let new_var = elite_vals
                        .iter()
                        .map(|&v| (v - new_mean).powi(2))
                        .sum::<f64>()
                        / n.max(1.0);
                    mean[t][d] = new_mean;
                    std[t][d] = new_var.sqrt().max(MIN_CEM_STD);
                }
            }
        }

        Ok(PlanResult {
            actions: best_actions,
            cost: best_cost,
            cost_history,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    #[should_panic(expected = "CostFunction::total_cost requires a non-empty trajectory")]
    fn test_l2_cost_empty_trajectory_panics() {
        let cost = L2Cost;
        let goal = Representation::<TestBackend>::new(Tensor::ones([1, 4, 8], &device()));
        let empty: Vec<Representation<TestBackend>> = vec![];
        let _ = cost.total_cost(&empty, &goal);
    }

    #[test]
    fn test_l2_cost_try_total_cost_returns_error_for_empty_trajectory() {
        let cost = L2Cost;
        let goal = Representation::<TestBackend>::new(Tensor::ones([1, 4, 8], &device()));
        let empty: Vec<Representation<TestBackend>> = vec![];

        let err = cost.try_total_cost(&empty, &goal).unwrap_err();
        assert_eq!(err, PlanningError::EmptyTrajectory);
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

    #[test]
    fn test_random_shooting_planner_finds_goal() {
        use rand::SeedableRng;
        let model = WorldModel::new(AdditiveDynamics, L2Cost);

        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let goal = Representation::new(Tensor::ones([1, 4, 8], &device()));

        let config = RandomShootingConfig {
            num_candidates: 128,
            num_iterations: 10,
            num_elites: 16,
            init_std: 2.0,
        };
        let planner = RandomShootingPlanner::new(config);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let result = planner.plan(&model, &initial, &goal, 1, 8, &mut rng);

        // The planner should find a plan with cost lower than the no-action baseline
        let baseline_cost: f32 = model
            .evaluate_plan(&initial, &[], &goal)
            .value
            .into_scalar()
            .elem();

        assert!(
            result.cost < baseline_cost,
            "planner should beat baseline: {} vs {}",
            result.cost,
            baseline_cost
        );
        assert_eq!(result.actions.len(), 1);
        assert_eq!(result.cost_history.len(), 10);
    }

    #[test]
    fn test_random_shooting_planner_cost_decreases() {
        use rand::SeedableRng;
        let model = WorldModel::new(AdditiveDynamics, L2Cost);

        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let goal = Representation::new(Tensor::ones([1, 4, 8], &device()));

        let config = RandomShootingConfig {
            num_candidates: 64,
            num_iterations: 5,
            num_elites: 8,
            init_std: 1.0,
        };
        let planner = RandomShootingPlanner::new(config);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(99);

        let result = planner.plan(&model, &initial, &goal, 2, 8, &mut rng);

        // Cost history should be monotonically non-increasing (best-so-far tracking)
        for w in result.cost_history.windows(2) {
            assert!(
                w[1] <= w[0],
                "cost history should be non-increasing: {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_random_shooting_planner_default_config() {
        let config = RandomShootingConfig::default();
        assert_eq!(config.num_candidates, 64);
        assert_eq!(config.num_iterations, 5);
        assert_eq!(config.num_elites, 8);
        assert!((config.init_std - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_random_shooting_config_validation() {
        let err = RandomShootingConfig {
            num_candidates: 0,
            num_iterations: 5,
            num_elites: 8,
            init_std: 1.0,
        }
        .validate()
        .unwrap_err();
        assert_eq!(err, PlanningError::ZeroCandidates(0));

        let err = RandomShootingConfig {
            num_candidates: 4,
            num_iterations: 0,
            num_elites: 1,
            init_std: 1.0,
        }
        .validate()
        .unwrap_err();
        assert_eq!(err, PlanningError::ZeroIterations(0));

        let err = RandomShootingConfig {
            num_candidates: 4,
            num_iterations: 1,
            num_elites: 0,
            init_std: 1.0,
        }
        .validate()
        .unwrap_err();
        assert_eq!(err, PlanningError::ZeroElites(0));
    }

    #[test]
    fn test_random_shooting_try_plan_rejects_zero_horizon() {
        use rand::SeedableRng;

        let model = WorldModel::new(AdditiveDynamics, L2Cost);
        let planner = RandomShootingPlanner::new(RandomShootingConfig::default());
        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let goal = Representation::new(Tensor::ones([1, 4, 8], &device()));
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);

        let err = planner
            .try_plan(&model, &initial, &goal, 0, 8, &mut rng)
            .unwrap_err();
        assert_eq!(err, PlanningError::ZeroHorizon(0));
    }

    #[test]
    fn test_random_shooting_try_plan_rejects_zero_action_dim() {
        use rand::SeedableRng;

        let model = WorldModel::new(AdditiveDynamics, L2Cost);
        let planner = RandomShootingPlanner::new(RandomShootingConfig::default());
        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let goal = Representation::new(Tensor::ones([1, 4, 8], &device()));
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);

        let err = planner
            .try_plan(&model, &initial, &goal, 1, 0, &mut rng)
            .unwrap_err();
        assert_eq!(err, PlanningError::ZeroActionDim(0));
    }

    #[test]
    fn test_cem_single_candidate() {
        use rand::SeedableRng;
        let model = WorldModel::new(AdditiveDynamics, L2Cost);
        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let goal = Representation::new(Tensor::ones([1, 4, 8], &device()));

        let config = RandomShootingConfig {
            num_candidates: 1,
            num_iterations: 3,
            num_elites: 1,
            init_std: 1.0,
        };
        let planner = RandomShootingPlanner::new(config);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let result = planner.plan(&model, &initial, &goal, 2, 8, &mut rng);
        assert_eq!(result.actions.len(), 2);
        assert!(result.cost.is_finite());
        assert_eq!(result.cost_history.len(), 3);
    }

    #[test]
    fn test_cem_elites_equal_candidates() {
        use rand::SeedableRng;
        let model = WorldModel::new(AdditiveDynamics, L2Cost);
        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let goal = Representation::new(Tensor::ones([1, 4, 8], &device()));

        // All candidates are elites — distribution refit uses all samples
        let config = RandomShootingConfig {
            num_candidates: 8,
            num_iterations: 3,
            num_elites: 8,
            init_std: 1.0,
        };
        let planner = RandomShootingPlanner::new(config);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let result = planner.plan(&model, &initial, &goal, 1, 8, &mut rng);
        assert!(result.cost.is_finite());
        assert_eq!(result.actions.len(), 1);
    }

    #[test]
    fn test_cem_elites_exceed_candidates() {
        use rand::SeedableRng;
        let model = WorldModel::new(AdditiveDynamics, L2Cost);
        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let goal = Representation::new(Tensor::ones([1, 4, 8], &device()));

        // num_elites > num_candidates — should be clamped gracefully
        let config = RandomShootingConfig {
            num_candidates: 4,
            num_iterations: 2,
            num_elites: 100,
            init_std: 1.0,
        };
        let planner = RandomShootingPlanner::new(config);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let result = planner.plan(&model, &initial, &goal, 1, 8, &mut rng);
        assert!(result.cost.is_finite());
    }

    #[test]
    fn test_cem_action_dim_one() {
        use rand::SeedableRng;
        let model = WorldModel::new(AdditiveDynamics, L2Cost);
        // Use embed_dim=1 so action_dim=1 matches for AdditiveDynamics
        let initial = Representation::new(Tensor::zeros([1, 4, 1], &device()));
        let goal = Representation::new(Tensor::ones([1, 4, 1], &device()));

        let config = RandomShootingConfig {
            num_candidates: 16,
            num_iterations: 3,
            num_elites: 4,
            init_std: 1.0,
        };
        let planner = RandomShootingPlanner::new(config);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        // Minimal action dimension
        let result = planner.plan(&model, &initial, &goal, 3, 1, &mut rng);
        assert_eq!(result.actions.len(), 3);
        for action in &result.actions {
            assert_eq!(action.action_dim(), 1);
        }
        assert!(result.cost.is_finite());
    }

    #[test]
    fn test_cem_single_iteration() {
        use rand::SeedableRng;
        let model = WorldModel::new(AdditiveDynamics, L2Cost);
        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let goal = Representation::new(Tensor::ones([1, 4, 8], &device()));

        let config = RandomShootingConfig {
            num_candidates: 32,
            num_iterations: 1,
            num_elites: 4,
            init_std: 1.0,
        };
        let planner = RandomShootingPlanner::new(config);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let result = planner.plan(&model, &initial, &goal, 2, 8, &mut rng);
        assert_eq!(result.cost_history.len(), 1);
        assert!(result.cost.is_finite());
    }

    #[test]
    fn test_cem_very_small_init_std() {
        use rand::SeedableRng;
        let model = WorldModel::new(AdditiveDynamics, L2Cost);
        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let goal = Representation::new(Tensor::ones([1, 4, 8], &device()));

        // Very small std — all candidates will be near mean (zero)
        let config = RandomShootingConfig {
            num_candidates: 32,
            num_iterations: 3,
            num_elites: 4,
            init_std: 1e-10,
        };
        let planner = RandomShootingPlanner::new(config);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let result = planner.plan(&model, &initial, &goal, 1, 8, &mut rng);
        assert!(result.cost.is_finite());
    }

    #[test]
    fn test_cem_large_init_std() {
        use rand::SeedableRng;
        let model = WorldModel::new(AdditiveDynamics, L2Cost);
        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let goal = Representation::new(Tensor::ones([1, 4, 8], &device()));

        // Very large std — wide exploration
        let config = RandomShootingConfig {
            num_candidates: 64,
            num_iterations: 5,
            num_elites: 8,
            init_std: 1000.0,
        };
        let planner = RandomShootingPlanner::new(config);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let result = planner.plan(&model, &initial, &goal, 1, 8, &mut rng);
        assert!(result.cost.is_finite());
    }

    #[test]
    fn test_cem_deterministic_with_same_seed() {
        use rand::SeedableRng;
        let model = WorldModel::new(AdditiveDynamics, L2Cost);
        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let goal = Representation::new(Tensor::ones([1, 4, 8], &device()));

        let config = RandomShootingConfig {
            num_candidates: 32,
            num_iterations: 3,
            num_elites: 4,
            init_std: 1.0,
        };

        let planner = RandomShootingPlanner::new(config.clone());
        let mut rng1 = rand_chacha::ChaCha8Rng::seed_from_u64(123);
        let result1 = planner.plan(&model, &initial, &goal, 2, 8, &mut rng1);

        let planner2 = RandomShootingPlanner::new(config);
        let mut rng2 = rand_chacha::ChaCha8Rng::seed_from_u64(123);
        let result2 = planner2.plan(&model, &initial, &goal, 2, 8, &mut rng2);

        assert_eq!(result1.cost, result2.cost);
        assert_eq!(result1.cost_history, result2.cost_history);
    }

    /// Dynamics model that produces NaN output for testing planner robustness.
    struct NanDynamics;

    impl ActionConditionedPredictor<TestBackend> for NanDynamics {
        fn predict_next_state(
            &self,
            current_state: &Representation<TestBackend>,
            _action: &Action<TestBackend>,
        ) -> Representation<TestBackend> {
            // Return NaN-filled tensor to simulate diverged dynamics
            let dims = current_state.embeddings.dims();
            let device = current_state.embeddings.device();
            Representation::new(Tensor::full(dims, f32::NAN, &device))
        }
    }

    #[test]
    fn test_cem_handles_nan_costs_without_panic() {
        use rand::SeedableRng;
        // When dynamics produce NaN, the cost function yields NaN.
        // The planner must not panic; it should still return a result.
        // Because every candidate has NaN cost, no candidate ever beats
        // the initial best_cost (f32::MAX), so best_actions stays empty.
        let model = WorldModel::new(NanDynamics, L2Cost);
        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let goal = Representation::new(Tensor::ones([1, 4, 8], &device()));

        let config = RandomShootingConfig {
            num_candidates: 8,
            num_iterations: 2,
            num_elites: 2,
            init_std: 1.0,
        };
        let planner = RandomShootingPlanner::new(config);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        // Must not panic — NaN costs are sorted deterministically via total_cmp
        let result = planner.plan(&model, &initial, &goal, 1, 8, &mut rng);
        // No valid plan found when all costs are NaN
        assert!(result.actions.is_empty());
        assert_eq!(result.cost_history.len(), 2);
        assert_eq!(result.cost, f32::MAX);
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
