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

// ---- Cross-crate integration: full JEPA training step ----

/// Integration test: full I-JEPA train step with real ViT encoder.
///
/// This validates the complete pipeline as described in RFC-008:
/// 1. Generate mask (jepa-core)
/// 2. Encode with real ViT (jepa-vision)
/// 3. Predict targets from context (jepa-vision)
/// 4. Compute energy + regularization (jepa-core)
/// 5. EMA update (jepa-core)
///
/// This test uses actual neural network modules (not stubs),
/// ensuring cross-crate compatibility.
#[test]
fn test_full_ijepa_train_step_with_real_vit() {
    use rand::SeedableRng;

    let config = IJepaConfig::tiny_test();
    let model = config.init::<TestBackend>(&device());

    // Create random input images: [batch=2, channels=1, height=8, width=8]
    let images: Tensor<TestBackend, 4> = Tensor::random(
        [2, 1, 8, 8],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device(),
    );

    // Step 1: Encode with both encoders (ViT forward pass)
    let context_repr = model.context_encoder.forward(&images);
    let target_repr = model.target_encoder.forward(&images);

    assert_eq!(context_repr.batch_size(), 2);
    assert_eq!(context_repr.seq_len(), 16); // 4x4 grid
    assert_eq!(context_repr.embed_dim(), 32);
    assert_eq!(target_repr.seq_len(), 16);

    // Step 2: Generate mask
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
    assert!(mask.validate().is_ok());

    // Step 3: Gather target tokens from target encoder output
    let target_gathered = target_repr.gather(&mask.target_indices);
    let num_targets = mask.target_indices.len();
    assert_eq!(target_gathered.batch_size(), 2);
    assert_eq!(target_gathered.seq_len(), num_targets);

    // Step 4: Predict targets from context using transformer predictor
    let target_pos: Tensor<TestBackend, 2> = Tensor::zeros([2, num_targets], &device());
    let predicted = model.predictor.predict(&context_repr, &target_pos, None);
    assert_eq!(predicted.batch_size(), 2);
    assert_eq!(predicted.seq_len(), num_targets);
    assert_eq!(predicted.embed_dim(), 32);

    // Step 5: Compute energy (prediction loss)
    let energy = jepa_core::energy::L2Energy.compute(&predicted, &target_gathered);
    let energy_val: f32 = energy.value.into_scalar().elem();
    assert!(
        energy_val.is_finite(),
        "energy should be finite: {energy_val}"
    );
    assert!(
        energy_val >= 0.0,
        "L2 energy should be non-negative: {energy_val}"
    );

    // Step 6: Compute collapse regularization (VICReg)
    let embed_dim = predicted.embed_dim();
    let batch = predicted.batch_size();
    let pred_flat = predicted
        .embeddings
        .clone()
        .reshape([batch * num_targets, embed_dim]);
    let target_flat = target_gathered
        .embeddings
        .clone()
        .reshape([batch * num_targets, embed_dim]);
    let vicreg = jepa_core::collapse::VICReg::default();
    let vicreg_loss = vicreg.compute(&pred_flat, &target_flat);
    let inv_val: f32 = vicreg_loss.invariance.into_scalar().elem();
    let var_val: f32 = vicreg_loss.variance.into_scalar().elem();
    let cov_val: f32 = vicreg_loss.covariance.into_scalar().elem();
    assert!(inv_val.is_finite(), "invariance loss should be finite");
    assert!(var_val.is_finite(), "variance loss should be finite");
    assert!(cov_val.is_finite(), "covariance loss should be finite");

    // Step 7: Verify total loss is computable
    let reg_weight = 1.0f32;
    let total_loss = energy_val + reg_weight * (inv_val + var_val + cov_val);
    assert!(
        total_loss.is_finite(),
        "total training loss should be finite: {total_loss}"
    );

    // Step 8: Simulate EMA update
    let ema = jepa_core::ema::Ema::new(0.996);
    let target_param: Tensor<TestBackend, 1> = Tensor::zeros([32], &device());
    let online_param: Tensor<TestBackend, 1> = Tensor::ones([32], &device());
    let updated = ema.update_tensor(target_param, &online_param, 0);
    let updated_val: f32 = updated.clone().into_data().to_vec::<f32>().unwrap()[0];
    assert!(
        (updated_val - 0.004).abs() < 1e-5,
        "EMA update should produce 0.004, got {updated_val}"
    );
}

/// Integration test: V-JEPA train step with spatiotemporal masking.
///
/// Validates the video pipeline end-to-end.
#[test]
fn test_full_vjepa_train_step_with_spatiotemporal_masking() {
    use jepa_vision::video::VJepaConfig;
    use rand::SeedableRng;

    let config = VJepaConfig::tiny_test();
    let model = config.init::<TestBackend>(&device());

    // Video input: [batch=1, channels=1, frames=4, height=8, width=8]
    let video: Tensor<TestBackend, 5> = Tensor::random(
        [1, 1, 4, 8, 8],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device(),
    );

    // Encode
    let context_repr = model.context_encoder.forward(&video);
    let target_repr = model.target_encoder.forward(&video);
    assert_eq!(context_repr.seq_len(), 32); // 2*4*4 tubelets

    // Spatiotemporal mask
    let masking = jepa_core::masking::SpatiotemporalMasking {
        num_targets: 2,
        temporal_extent: (1, 2),
        spatial_scale: (0.1, 0.2),
    };
    let shape = InputShape::Video {
        frames: 2,
        height: 4,
        width: 4,
    };
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let mask = masking.generate_mask(&shape, &mut rng);
    assert!(mask.validate().is_ok());

    // Gather and predict
    let target_gathered = target_repr.gather(&mask.target_indices);
    let num_targets = mask.target_indices.len();
    let target_pos: Tensor<TestBackend, 2> = Tensor::zeros([1, num_targets], &device());
    let predicted = model.predictor.predict(&context_repr, &target_pos, None);

    // Energy
    let energy = jepa_core::energy::L2Energy.compute(&predicted, &target_gathered);
    let energy_val: f32 = energy.value.into_scalar().elem();
    assert!(energy_val.is_finite(), "V-JEPA energy should be finite");
    assert!(energy_val >= 0.0, "V-JEPA energy should be non-negative");

    // Cosine energy as alternative
    let cosine_energy = jepa_core::energy::CosineEnergy.compute(&predicted, &target_gathered);
    let cosine_val: f32 = cosine_energy.value.into_scalar().elem();
    assert!(cosine_val.is_finite(), "cosine energy should be finite");

    // Barlow Twins regularization
    let embed_dim = predicted.embed_dim();
    let pred_flat = predicted.embeddings.reshape([num_targets, embed_dim]);
    let target_flat = target_gathered.embeddings.reshape([num_targets, embed_dim]);
    let bt = jepa_core::collapse::BarlowTwins::default();
    let bt_loss = bt.compute(&pred_flat, &target_flat);
    let bt_total: f32 = bt_loss.total().into_scalar().elem();
    assert!(bt_total.is_finite(), "Barlow Twins loss should be finite");
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
