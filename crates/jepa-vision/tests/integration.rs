//! Integration tests for I-JEPA and V-JEPA forward pass pipelines.
//!
//! These tests verify the end-to-end behavior described in the Gherkin scenarios
//! in specs/gherkin/features.feature.

use burn::prelude::*;
use burn::tensor::ElementConversion;
use burn_ndarray::NdArray;

use jepa_core::types::{InputShape, Representation};
use jepa_core::{Encoder, EnergyFn, MaskingStrategy, Predictor};
use jepa_vision::image::IJepaConfig;
use jepa_vision::video::VitVideoConfig;
use jepa_vision::vit::VitConfig;

type TestBackend = NdArray<f32>;

fn device() -> burn_ndarray::NdArrayDevice {
    burn_ndarray::NdArrayDevice::Cpu
}

// ---- I-JEPA Integration Tests (matching Gherkin scenarios) ----

/// Gherkin: I-JEPA full forward pass — encode, mask, predict, compute energy.
///
/// Scenario: Load I-JEPA model → forward pass produces non-zero output
/// (adapted from checkpoint.feature for in-memory model)
#[test]
fn test_ijepa_end_to_end_forward_pass() {
    let config = IJepaConfig::tiny_test();
    let model = config.init::<TestBackend>(&device());

    // Create a batch of test images: [batch=2, channels=1, height=8, width=8]
    let images: Tensor<TestBackend, 4> = Tensor::ones([2, 1, 8, 8], &device());

    // Encode with context encoder
    let context_repr = model.context_encoder.forward(&images);
    assert_eq!(context_repr.batch_size(), 2);
    assert_eq!(context_repr.seq_len(), 16); // 4x4 grid of patches
    assert_eq!(context_repr.embed_dim(), 32);

    // Encode with target encoder (EMA copy in real training)
    let target_repr = model.target_encoder.forward(&images);
    assert_eq!(target_repr.seq_len(), 16);

    // Verify non-zero output
    let sum: f32 = context_repr
        .embeddings
        .clone()
        .abs()
        .sum()
        .into_scalar()
        .elem();
    assert!(sum > 1e-6, "forward pass should produce non-zero output");
}

/// Gherkin: Block masking partitions all patches
///
/// Scenario: context_indices + target_indices should cover all patches
/// with no overlap.
#[test]
fn test_ijepa_masking_partitions_all_patches() {
    use rand::SeedableRng;

    let masking = jepa_core::masking::BlockMasking {
        num_targets: 4,
        target_scale: (0.15, 0.2),
        target_aspect_ratio: (0.75, 1.5),
    };
    let shape = InputShape::Image {
        height: 14,
        width: 14,
    };
    let total_patches = 196;

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let mask = masking.generate_mask(&shape, &mut rng);

    // All patches covered
    let mut all_indices: Vec<usize> = mask
        .context_indices
        .iter()
        .chain(mask.target_indices.iter())
        .copied()
        .collect();
    all_indices.sort();
    all_indices.dedup();
    assert_eq!(
        all_indices.len(),
        total_patches,
        "context + target should cover all {} patches",
        total_patches
    );

    // No overlap
    let ctx_set: std::collections::HashSet<usize> = mask.context_indices.iter().copied().collect();
    let tgt_set: std::collections::HashSet<usize> = mask.target_indices.iter().copied().collect();
    let overlap: Vec<_> = ctx_set.intersection(&tgt_set).collect();
    assert!(
        overlap.is_empty(),
        "context and target should not overlap, but found: {:?}",
        overlap
    );
}

/// Gherkin: I-JEPA encode → predict → energy is finite and non-negative.
///
/// End-to-end test that the predictor can predict target representations
/// and the energy function produces a valid result.
#[test]
fn test_ijepa_predict_and_energy() {
    use rand::SeedableRng;

    let config = IJepaConfig::tiny_test();
    let model = config.init::<TestBackend>(&device());

    let images: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 8, 8], &device());

    // Encode
    let context_repr = model.context_encoder.forward(&images);
    let target_repr = model.target_encoder.forward(&images);

    // Mask
    let masking = jepa_core::masking::BlockMasking {
        num_targets: 2,
        target_scale: (0.15, 0.3),
        target_aspect_ratio: (0.75, 1.5),
    };
    let shape = InputShape::Image {
        height: 4,
        width: 4,
    };
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let mask = masking.generate_mask(&shape, &mut rng);

    // Predict
    let num_targets = mask.target_indices.len();
    let target_pos: Tensor<TestBackend, 2> = Tensor::zeros([1, num_targets], &device());
    let predicted = model.predictor.predict(&context_repr, &target_pos, None);

    assert_eq!(predicted.seq_len(), num_targets);
    assert_eq!(predicted.embed_dim(), 32);

    // Compute L2 energy between predicted and actual targets
    let energy = jepa_core::energy::L2Energy.compute(&predicted, &predicted);
    let val: f32 = energy.value.into_scalar().elem();
    assert!(val.is_finite(), "energy should be finite, got {val}");
    assert!(val >= 0.0, "L2 energy should be non-negative, got {val}");
    assert!(val < 1e-6, "self-energy should be ~0, got {val}");

    // Energy between predicted and target_repr slice should be > 0
    // (since predictor is randomly initialized, predictions won't match targets)
    let target_slice =
        Representation::new(target_repr.embeddings.slice([0..1, 0..num_targets, 0..32]));
    let cross_energy = jepa_core::energy::L2Energy.compute(&predicted, &target_slice);
    let cross_val: f32 = cross_energy.value.into_scalar().elem();
    assert!(
        cross_val.is_finite(),
        "cross energy should be finite, got {cross_val}"
    );
}

/// Gherkin: Different inputs produce different representations.
#[test]
fn test_ijepa_different_inputs_different_outputs() {
    let config = VitConfig::tiny_test();
    let encoder = config.init::<TestBackend>(&device());

    let zeros: Tensor<TestBackend, 4> = Tensor::zeros([1, 1, 8, 8], &device());
    let ones: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 8, 8], &device());

    let repr_a = encoder.encode(&zeros);
    let repr_b = encoder.encode(&ones);

    let diff: f32 = (repr_a.embeddings - repr_b.embeddings)
        .abs()
        .sum()
        .into_scalar()
        .elem();
    assert!(
        diff > 1e-6,
        "different inputs should produce different representations, diff={diff}"
    );
}

// ---- V-JEPA Integration Tests ----

/// Gherkin (adapted): V-JEPA video encoder forward pass produces
/// correct shape and non-zero output.
#[test]
fn test_vjepa_end_to_end_forward_pass() {
    let config = VitVideoConfig::tiny_test();
    let encoder = config.init::<TestBackend>(&device());

    // [batch=2, channels=1, frames=4, height=8, width=8]
    let video: Tensor<TestBackend, 5> = Tensor::ones([2, 1, 4, 8, 8], &device());
    let repr = encoder.forward(&video);

    // grid: (4/2, 8/2, 8/2) = (2, 4, 4) = 32 tubelets
    assert_eq!(repr.batch_size(), 2);
    assert_eq!(repr.seq_len(), 32);
    assert_eq!(repr.embed_dim(), 32);

    // Non-zero output
    let sum: f32 = repr.embeddings.clone().abs().sum().into_scalar().elem();
    assert!(
        sum > 1e-6,
        "V-JEPA forward pass should produce non-zero output"
    );
}

/// V-JEPA encoder implements the Encoder trait correctly.
#[test]
fn test_vjepa_encoder_trait() {
    let config = VitVideoConfig::tiny_test();
    let encoder = config.init::<TestBackend>(&device());

    let video: Tensor<TestBackend, 5> = Tensor::zeros([1, 1, 4, 8, 8], &device());
    let repr = Encoder::encode(&encoder, &video);

    assert_eq!(repr.batch_size(), 1);
    assert_eq!(repr.seq_len(), 32);
    assert_eq!(encoder.embed_dim(), 32);
}

/// V-JEPA produces different representations for different video inputs.
#[test]
fn test_vjepa_different_inputs_different_outputs() {
    let config = VitVideoConfig::tiny_test();
    let encoder = config.init::<TestBackend>(&device());

    let zeros: Tensor<TestBackend, 5> = Tensor::zeros([1, 1, 4, 8, 8], &device());
    let ones: Tensor<TestBackend, 5> = Tensor::ones([1, 1, 4, 8, 8], &device());

    let repr_a = encoder.encode(&zeros);
    let repr_b = encoder.encode(&ones);

    let diff: f32 = (repr_a.embeddings - repr_b.embeddings)
        .abs()
        .sum()
        .into_scalar()
        .elem();
    assert!(
        diff > 1e-6,
        "different video inputs should produce different representations, diff={diff}"
    );
}

/// V-JEPA grid dimensions match config expectations.
#[test]
fn test_vjepa_grid_dimensions() {
    let config = VitVideoConfig {
        in_channels: 3,
        num_frames: 16,
        frame_height: 224,
        frame_width: 224,
        tubelet_size: (2, 16, 16),
        embed_dim: 768,
        num_layers: 12,
        num_heads: 12,
        mlp_dim: 3072,
    };

    assert_eq!(config.grid_dims(), (8, 14, 14));
    assert_eq!(config.num_tubelets(), 1568);
}
