//! Criterion benchmarks for jepa-world operations.
//!
//! Run with: `cargo bench -p jepa-world`

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use burn::prelude::*;
use burn_ndarray::NdArray;

use jepa_core::types::Representation;
use jepa_world::action::{Action, ActionConditionedPredictor};
use jepa_world::planner::{L2Cost, RandomShootingConfig, RandomShootingPlanner, WorldModel};

use rand::SeedableRng;

type B = NdArray<f32>;

fn device() -> burn_ndarray::NdArrayDevice {
    burn_ndarray::NdArrayDevice::Cpu
}

/// Simple additive dynamics for benchmarking.
struct AdditiveDynamics;

impl ActionConditionedPredictor<B> for AdditiveDynamics {
    fn predict_next_state(
        &self,
        current_state: &Representation<B>,
        action: &Action<B>,
    ) -> Representation<B> {
        let [batch, seq_len, embed_dim] = current_state.embeddings.dims();
        let a = action
            .data
            .clone()
            .slice([0..batch, 0..embed_dim.min(action.action_dim())])
            .unsqueeze::<3>()
            .expand([batch, seq_len, embed_dim]);
        Representation::new(current_state.embeddings.clone() + a)
    }
}

// --- Rollout Benchmarks ---

fn bench_rollout(c: &mut Criterion) {
    let mut group = c.benchmark_group("world/rollout");

    for &num_steps in &[1usize, 5, 20] {
        let model = WorldModel::new(AdditiveDynamics, L2Cost);
        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let actions: Vec<Action<B>> = (0..num_steps)
            .map(|_| Action::new(Tensor::ones([1, 8], &device())))
            .collect();

        group.bench_with_input(BenchmarkId::new("steps", num_steps), &num_steps, |b, _| {
            b.iter(|| {
                black_box(model.rollout(&initial, &actions));
            })
        });
    }

    group.finish();
}

// --- CEM Planner Benchmarks ---

fn bench_cem_planner(c: &mut Criterion) {
    let mut group = c.benchmark_group("world/cem_planner");

    for &num_candidates in &[16usize, 64, 256] {
        let model = WorldModel::new(AdditiveDynamics, L2Cost);
        let initial = Representation::new(Tensor::zeros([1, 4, 8], &device()));
        let goal = Representation::new(Tensor::ones([1, 4, 8], &device()));

        let config = RandomShootingConfig {
            num_candidates,
            num_iterations: 3,
            num_elites: (num_candidates / 8).max(1),
            init_std: 1.0,
        };
        let planner = RandomShootingPlanner::new(config);

        group.bench_with_input(
            BenchmarkId::new("candidates", num_candidates),
            &num_candidates,
            |b, _| {
                b.iter(|| {
                    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
                    black_box(planner.plan(&model, &initial, &goal, 3, 8, &mut rng));
                })
            },
        );
    }

    group.finish();
}

// --- L2 Cost Benchmarks ---

fn bench_l2_cost(c: &mut Criterion) {
    use jepa_world::planner::CostFunction;

    let mut group = c.benchmark_group("world/l2_cost");

    for &(seq_len, embed_dim) in &[(4, 8), (16, 64), (64, 256)] {
        let cost = L2Cost;
        let trajectory: Vec<Representation<B>> = (0..5)
            .map(|_| {
                Representation::new(Tensor::random(
                    [1, seq_len, embed_dim],
                    burn::tensor::Distribution::Normal(0.0, 1.0),
                    &device(),
                ))
            })
            .collect();
        let goal = Representation::new(Tensor::random(
            [1, seq_len, embed_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        ));

        group.bench_with_input(
            BenchmarkId::new("dims", format!("{seq_len}x{embed_dim}")),
            &(seq_len, embed_dim),
            |b, _| {
                b.iter(|| {
                    black_box(cost.total_cost(&trajectory, &goal));
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_rollout, bench_cem_planner, bench_l2_cost,);
criterion_main!(benches);
