//! Criterion benchmarks for jepa-vision operations.
//!
//! Run with: `cargo bench -p jepa-vision`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use burn::prelude::*;
use burn_ndarray::NdArray;

use jepa_core::types::Representation;
use jepa_core::Predictor;
use jepa_vision::image::TransformerPredictorConfig;
use jepa_vision::patch::PatchEmbeddingConfig;
use jepa_vision::vit::VitConfig;

type B = NdArray<f32>;

fn device() -> burn_ndarray::NdArrayDevice {
    burn_ndarray::NdArrayDevice::Cpu
}

// --- Patch Embedding Benchmarks ---

fn bench_patch_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("vision/patch_embed");

    for &(channels, patch_size, img_size, embed_dim) in
        &[(1, 2, 8, 32), (3, 16, 64, 256), (3, 16, 224, 768)]
    {
        let config = PatchEmbeddingConfig::new(channels, patch_size, patch_size, embed_dim);
        let pe = config.init::<B>(&device());
        let images: Tensor<B, 4> = Tensor::zeros([1, channels, img_size, img_size], &device());

        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "{img_size}x{img_size}_p{patch_size}_d{embed_dim}"
            )),
            &(pe, images),
            |bencher, (pe, images)| {
                bencher.iter(|| {
                    let out = pe.forward(black_box(images.clone()));
                    black_box(out);
                });
            },
        );
    }

    group.finish();
}

// --- ViT Encoder Benchmarks ---

fn bench_vit_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("vision/vit_encoder");

    // Tiny config for fast benchmarking
    let config = VitConfig::tiny_test();
    let encoder = config.init::<B>(&device());

    for &batch in &[1, 2, 4] {
        let images: Tensor<B, 4> = Tensor::zeros([batch, 1, 8, 8], &device());

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("tiny_b{batch}")),
            &images,
            |bencher, images| {
                bencher.iter(|| {
                    let repr = encoder.forward(black_box(images));
                    black_box(repr);
                });
            },
        );
    }

    group.finish();
}

// --- Predictor Benchmarks ---

fn bench_predictor(c: &mut Criterion) {
    let mut group = c.benchmark_group("vision/predictor");

    for &(embed_dim, pred_dim, num_targets) in &[(32, 16, 4), (32, 16, 8), (64, 32, 16)] {
        let config = TransformerPredictorConfig {
            encoder_embed_dim: embed_dim,
            predictor_embed_dim: pred_dim,
            num_layers: 1,
            num_heads: 2,
            max_target_len: 64,
        };
        let predictor = config.init::<B>(&device());
        let context = Representation::new(Tensor::zeros([1, 8, embed_dim], &device()));
        let target_pos: Tensor<B, 2> = Tensor::zeros([1, num_targets], &device());

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d{embed_dim}_pd{pred_dim}_t{num_targets}")),
            &(context, target_pos),
            |bencher, (ctx, tpos)| {
                bencher.iter(|| {
                    let pred = predictor.predict(black_box(ctx), black_box(tpos), None);
                    black_box(pred);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_patch_embedding,
    bench_vit_encoder,
    bench_predictor,
);
criterion_main!(benches);
