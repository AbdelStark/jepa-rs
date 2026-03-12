//! Criterion benchmarks for jepa-core operations.
//!
//! Run with: `cargo bench -p jepa-core`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use burn::prelude::*;
use burn_ndarray::NdArray;

use jepa_core::collapse::VICReg;
use jepa_core::ema::Ema;
use jepa_core::energy::{CosineEnergy, EnergyFn, L2Energy, SmoothL1Energy};
use jepa_core::masking::{BlockMasking, MaskingStrategy};
use jepa_core::types::{InputShape, Representation};

use rand::SeedableRng;

type B = NdArray<f32>;

fn device() -> burn_ndarray::NdArrayDevice {
    burn_ndarray::NdArrayDevice::Cpu
}

fn make_repr(batch: usize, seq_len: usize, embed_dim: usize) -> Representation<B> {
    Representation::new(Tensor::random(
        [batch, seq_len, embed_dim],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device(),
    ))
}

// --- Energy Function Benchmarks ---

fn bench_l2_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("energy/l2");

    for &(batch, seq_len, embed_dim) in &[(1, 16, 64), (4, 64, 256), (8, 196, 768)] {
        let a = make_repr(batch, seq_len, embed_dim);
        let b = make_repr(batch, seq_len, embed_dim);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("b{batch}_s{seq_len}_d{embed_dim}")),
            &(a, b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    let energy = L2Energy.compute(black_box(a), black_box(b));
                    black_box(energy);
                });
            },
        );
    }

    group.finish();
}

fn bench_cosine_energy(c: &mut Criterion) {
    let mut group = c.benchmark_group("energy/cosine");

    for &(batch, seq_len, embed_dim) in &[(1, 16, 64), (4, 64, 256), (8, 196, 768)] {
        let a = make_repr(batch, seq_len, embed_dim);
        let b = make_repr(batch, seq_len, embed_dim);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("b{batch}_s{seq_len}_d{embed_dim}")),
            &(a, b),
            |bencher, (a, b)| {
                bencher.iter(|| {
                    let energy = CosineEnergy.compute(black_box(a), black_box(b));
                    black_box(energy);
                });
            },
        );
    }

    group.finish();
}

fn bench_smooth_l1_energy(c: &mut Criterion) {
    let a = make_repr(4, 64, 256);
    let b = make_repr(4, 64, 256);
    let energy_fn = SmoothL1Energy::new(1.0);

    c.bench_function("energy/smooth_l1/b4_s64_d256", |bencher| {
        bencher.iter(|| {
            let energy = energy_fn.compute(black_box(&a), black_box(&b));
            black_box(energy);
        });
    });
}

// --- Masking Benchmarks ---

fn bench_block_masking(c: &mut Criterion) {
    let mut group = c.benchmark_group("masking/block");

    for &(height, width) in &[(8, 8), (14, 14), (16, 16)] {
        let masking = BlockMasking {
            num_targets: 4,
            target_scale: (0.15, 0.2),
            target_aspect_ratio: (0.75, 1.5),
        };
        let shape = InputShape::Image { height, width };

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{height}x{width}")),
            &(masking, shape),
            |bencher, (masking, shape)| {
                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
                bencher.iter(|| {
                    let mask = masking.generate_mask(black_box(shape), &mut rng);
                    black_box(mask);
                });
            },
        );
    }

    group.finish();
}

// --- EMA Benchmarks ---

fn bench_ema_scalar(c: &mut Criterion) {
    let ema = Ema::new(0.996);

    c.bench_function("ema/scalar_step", |bencher| {
        bencher.iter(|| {
            let result = ema.step(black_box(0.5), black_box(1.0), black_box(100));
            black_box(result);
        });
    });
}

fn bench_ema_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("ema/tensor");

    for &dim in &[64, 256, 768] {
        let ema = Ema::new(0.996);
        let target: Tensor<B, 2> = Tensor::zeros([dim, dim], &device());
        let online: Tensor<B, 2> = Tensor::ones([dim, dim], &device());

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{dim}x{dim}")),
            &(target, online),
            |bencher, (target, online)| {
                bencher.iter(|| {
                    let result = ema.update_tensor(
                        black_box(target.clone()),
                        black_box(online),
                        black_box(0),
                    );
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

// --- VICReg Benchmarks ---

fn bench_vicreg(c: &mut Criterion) {
    let mut group = c.benchmark_group("vicreg");

    for &(batch, embed_dim) in &[(16, 64), (32, 256), (64, 768)] {
        let vicreg = VICReg::default();
        let z_a: Tensor<B, 2> = Tensor::random(
            [batch, embed_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let z_b: Tensor<B, 2> = Tensor::random(
            [batch, embed_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("b{batch}_d{embed_dim}")),
            &(z_a, z_b),
            |bencher, (z_a, z_b)| {
                bencher.iter(|| {
                    let loss = vicreg.compute(black_box(z_a), black_box(z_b));
                    black_box(loss);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_l2_energy,
    bench_cosine_energy,
    bench_smooth_l1_energy,
    bench_block_masking,
    bench_ema_scalar,
    bench_ema_tensor,
    bench_vicreg,
);
criterion_main!(benches);
