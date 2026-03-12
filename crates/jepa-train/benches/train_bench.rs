//! Criterion benchmarks for jepa-train operations.
//!
//! Run with: `cargo bench -p jepa-train`

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use burn::prelude::*;
use burn_ndarray::NdArray;
use rand::SeedableRng;

use jepa_core::collapse::VICReg;
use jepa_core::energy::L2Energy;
use jepa_core::masking::BlockMasking;
use jepa_core::types::{InputShape, Representation};
use jepa_core::{Encoder, Predictor};
use jepa_train::schedule::{LrSchedule, WarmupCosineSchedule};
use jepa_train::trainer::JepaComponents;

type B = NdArray<f32>;

fn device() -> burn_ndarray::NdArrayDevice {
    burn_ndarray::NdArrayDevice::Cpu
}

// --- Test helpers ---

#[derive(Clone)]
struct BenchEncoder {
    embed_dim: usize,
}

impl Encoder<B> for BenchEncoder {
    type Input = Tensor<B, 4>;

    fn encode(&self, input: &Self::Input) -> Representation<B> {
        let [batch, _c, h, w] = input.dims();
        let seq_len = h * w;
        Representation::new(Tensor::ones(
            [batch, seq_len, self.embed_dim],
            &input.device(),
        ))
    }

    fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

struct BenchPredictor {
    embed_dim: usize,
}

impl Predictor<B> for BenchPredictor {
    fn predict(
        &self,
        _context: &Representation<B>,
        target_positions: &Tensor<B, 2>,
        _latent: Option<&Tensor<B, 2>>,
    ) -> Representation<B> {
        let [batch, num_targets] = target_positions.dims();
        Representation::new(Tensor::ones(
            [batch, num_targets, self.embed_dim],
            &target_positions.device(),
        ))
    }
}

// --- LR Schedule Benchmarks ---

fn bench_warmup_cosine_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("schedule/warmup_cosine");

    for &total_steps in &[1_000usize, 10_000, 100_000] {
        let schedule = WarmupCosineSchedule::new(1e-3, total_steps / 10, total_steps);

        group.bench_with_input(
            BenchmarkId::new("single_step", total_steps),
            &total_steps,
            |b, &total| {
                b.iter(|| {
                    black_box(schedule.get_lr(total / 2));
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("full_sweep", total_steps),
            &total_steps,
            |b, &total| {
                b.iter(|| {
                    for step in (0..total).step_by(total / 100) {
                        black_box(schedule.get_lr(step));
                    }
                })
            },
        );
    }

    group.finish();
}

// --- JepaComponents Forward Step Benchmarks ---

fn bench_jepa_forward_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("trainer/forward_step");

    for &embed_dim in &[16, 64, 128] {
        let encoder = BenchEncoder { embed_dim };
        let predictor = BenchPredictor { embed_dim };
        let energy_fn = L2Energy;
        let regularizer = VICReg::default();
        let masking = BlockMasking {
            num_targets: 2,
            target_scale: (0.15, 0.3),
            target_aspect_ratio: (0.75, 1.5),
        };

        let input: Tensor<B, 4> = Tensor::ones([2, 1, 4, 4], &device());
        let input_shape = InputShape::Image {
            height: 4,
            width: 4,
        };

        let components = JepaComponents::new(
            &encoder,
            &encoder,
            &predictor,
            &energy_fn,
            &regularizer,
            &masking,
            1.0,
        );

        group.bench_with_input(
            BenchmarkId::new("embed_dim", embed_dim),
            &embed_dim,
            |b, _| {
                b.iter(|| {
                    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
                    black_box(components.forward_step(&input, &input_shape, &mut rng));
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_warmup_cosine_schedule,
    bench_jepa_forward_step,
);
criterion_main!(benches);
