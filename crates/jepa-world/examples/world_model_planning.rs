//! World model planning demonstration.
//!
//! This example demonstrates the action-conditioned world model and
//! random-shooting planner (CEM) from RFC-009.
//!
//! Scenario: A simple 2D navigation task where an agent must plan
//! a sequence of actions to move from a start position to a goal
//! position in representation space.
//!
//! Run with: `cargo run -p jepa-world --example world_model_planning`

use burn::tensor::{ElementConversion, Tensor};
use burn_ndarray::NdArray;
use rand::SeedableRng;

use jepa_core::types::Representation;
use jepa_world::action::{Action, ActionConditionedPredictor};
use jepa_world::planner::{L2Cost, RandomShootingConfig, RandomShootingPlanner, WorldModel};

type B = NdArray<f32>;

/// Simple linear dynamics model for the 2D navigation task.
///
/// The state evolves as: next_state = state + action
/// This models a simple integrator (position += velocity).
struct NavigationDynamics;

impl ActionConditionedPredictor<B> for NavigationDynamics {
    fn predict_next_state(
        &self,
        current_state: &Representation<B>,
        action: &Action<B>,
    ) -> Representation<B> {
        let [batch, seq_len, embed_dim] = current_state.embeddings.dims();
        let action_dim = action.action_dim();
        let dim = embed_dim.min(action_dim);

        // Broadcast action across sequence dimension
        let a = action
            .data
            .clone()
            .slice([0..batch, 0..dim])
            .unsqueeze::<3>()
            .expand([batch, seq_len, dim]);

        // For dims beyond action_dim, state stays the same
        if dim < embed_dim {
            let unchanged = current_state
                .embeddings
                .clone()
                .slice([0..batch, 0..seq_len, dim..embed_dim]);
            let changed = current_state
                .embeddings
                .clone()
                .slice([0..batch, 0..seq_len, 0..dim])
                + a;
            Representation::new(Tensor::cat(vec![changed, unchanged], 2))
        } else {
            Representation::new(current_state.embeddings.clone() + a)
        }
    }
}

fn main() {
    println!("=== World Model Planning Demo ===\n");

    let device = burn_ndarray::NdArrayDevice::Cpu;

    // --- Setup ---
    let embed_dim = 4;
    let action_dim = 4;
    let seq_len = 1;

    // Start at the origin
    let start = Representation::new(Tensor::<B, 3>::zeros([1, seq_len, embed_dim], &device));
    // Goal: [1, 1, 1, 1]
    let goal = Representation::new(Tensor::<B, 3>::ones([1, seq_len, embed_dim], &device));

    println!("Navigation task:");
    println!("  State dim: {embed_dim}, Action dim: {action_dim}");
    println!("  Start: [0, 0, 0, 0]");
    println!("  Goal:  [1, 1, 1, 1]");
    println!();

    // --- Create World Model ---
    let world_model = WorldModel::new(NavigationDynamics, L2Cost);

    // --- Baseline: no action ---
    let baseline_cost: f32 = world_model
        .evaluate_plan(&start, &[], &goal)
        .value
        .into_scalar()
        .elem();
    println!("Baseline cost (no action): {:.6}", baseline_cost);
    println!();

    // --- Manual plan: single step to goal ---
    let perfect_plan = vec![Action::new(Tensor::ones([1, action_dim], &device))];
    let perfect_cost: f32 = world_model
        .evaluate_plan(&start, &perfect_plan, &goal)
        .value
        .into_scalar()
        .elem();
    println!("Perfect plan cost (oracle): {:.6}", perfect_cost);

    // --- Rollout visualization ---
    let trajectory = world_model.rollout(&start, &perfect_plan);
    println!("Rollout ({} states):", trajectory.len());
    for (i, state) in trajectory.iter().enumerate() {
        let vals: Vec<f32> = state.embeddings.clone().reshape([embed_dim]).into_data().to_vec().unwrap();
        println!(
            "  t={}: [{:.2}, {:.2}, {:.2}, {:.2}]",
            i, vals[0], vals[1], vals[2], vals[3]
        );
    }
    println!();

    // --- CEM Planning ---
    println!("Running CEM planner...");
    let config = RandomShootingConfig {
        num_candidates: 256,
        num_iterations: 10,
        num_elites: 32,
        init_std: 2.0,
    };
    let planner = RandomShootingPlanner::new(config.clone());
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

    // Plan a multi-step trajectory
    let horizon = 3;
    let result = planner.plan(&world_model, &start, &goal, horizon, action_dim, &mut rng);

    println!("CEM Planner config:");
    println!("  Candidates: {}", config.num_candidates);
    println!("  Iterations: {}", config.num_iterations);
    println!("  Elites:     {}", config.num_elites);
    println!("  Horizon:    {horizon}");
    println!();

    println!("Cost history:");
    for (i, &cost) in result.cost_history.iter().enumerate() {
        let bar_len = (50.0 * cost / baseline_cost).min(50.0) as usize;
        let bar: String = "#".repeat(bar_len);
        println!("  iter {:>2}: cost={:.6} {}", i, cost, bar);
    }
    println!();

    println!("Best plan (cost={:.6}):", result.cost);
    let planned_trajectory = world_model.rollout(&start, &result.actions);
    for (i, state) in planned_trajectory.iter().enumerate() {
        let vals: Vec<f32> = state.embeddings.clone().reshape([embed_dim]).into_data().to_vec().unwrap();
        println!(
            "  t={}: [{:.2}, {:.2}, {:.2}, {:.2}]",
            i, vals[0], vals[1], vals[2], vals[3]
        );
    }
    println!();

    // --- Summary ---
    let improvement = (1.0 - result.cost / baseline_cost) * 100.0;
    println!("Summary:");
    println!("  Baseline cost:  {:.6}", baseline_cost);
    println!("  Planned cost:   {:.6}", result.cost);
    println!("  Improvement:    {:.1}%", improvement);
    println!("  Plan length:    {} actions", result.actions.len());

    assert!(
        result.cost < baseline_cost,
        "planner should improve over baseline"
    );
    println!();
    println!("=== Demo Complete ===");
}
