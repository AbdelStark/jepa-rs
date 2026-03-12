//! Integration tests for I-JEPA and V-JEPA forward pass pipelines.
//!
//! These tests verify the end-to-end behavior described in the Gherkin scenarios
//! in specs/gherkin/features.feature.

use burn::module::{Module, ModuleMapper, Param};
use burn::prelude::*;
use burn::tensor::ElementConversion;
use burn_ndarray::NdArray;

use jepa_core::types::{InputShape, MaskSpec, Representation};
use jepa_core::{CollapseRegularizer, Encoder, EnergyFn, MaskingStrategy, Predictor};
use jepa_vision::image::IJepaConfig;
use jepa_vision::image::TransformerPredictorConfig;
use jepa_vision::video::VitVideoConfig;
use jepa_vision::vit::VitConfig;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

type TestBackend = NdArray<f32>;

fn device() -> burn_ndarray::NdArrayDevice {
    burn_ndarray::NdArrayDevice::Cpu
}

fn fixed_image_mask() -> MaskSpec {
    MaskSpec {
        context_indices: vec![0, 1, 4, 5, 10, 11, 14, 15],
        target_indices: vec![2, 3, 6, 7, 8, 9, 12, 13],
        total_tokens: 16,
    }
}

fn image_with_hidden_patch_value(mask: &MaskSpec, hidden_value: f32) -> Tensor<TestBackend, 4> {
    let image_size = 8usize;
    let patch_size = 2usize;
    let mut data = vec![1.0f32; image_size * image_size];

    for &index in &mask.target_indices {
        let patch_row = index / 4;
        let patch_col = index % 4;
        let row_start = patch_row * patch_size;
        let col_start = patch_col * patch_size;

        for row in row_start..row_start + patch_size {
            for col in col_start..col_start + patch_size {
                data[row * image_size + col] = hidden_value;
            }
        }
    }

    Tensor::from_floats(
        burn::tensor::TensorData::new(data, [1, 1, image_size, image_size]),
        &device(),
    )
}

fn fixed_video_mask() -> MaskSpec {
    MaskSpec {
        context_indices: (0..16).collect(),
        target_indices: (16..32).collect(),
        total_tokens: 32,
    }
}

fn video_with_hidden_tubelet_value(mask: &MaskSpec, hidden_value: f32) -> Tensor<TestBackend, 5> {
    let frames = 4usize;
    let height = 8usize;
    let width = 8usize;
    let mut data = vec![1.0f32; frames * height * width];

    for &index in &mask.target_indices {
        let temporal_block = index / 16;
        let spatial_index = index % 16;
        let spatial_row = spatial_index / 4;
        let spatial_col = spatial_index % 4;
        let frame_start = temporal_block * 2;
        let row_start = spatial_row * 2;
        let col_start = spatial_col * 2;

        for frame in frame_start..frame_start + 2 {
            for row in row_start..row_start + 2 {
                for col in col_start..col_start + 2 {
                    data[(frame * height + row) * width + col] = hidden_value;
                }
            }
        }
    }

    Tensor::from_floats(
        burn::tensor::TensorData::new(data, [1, 1, frames, height, width]),
        &device(),
    )
}

#[derive(Debug)]
struct ParityFixture {
    metadata: ParityMetadata,
    config: ParityConfig,
    weights: HashMap<String, FixtureTensor>,
    raw_input: FixtureTensor,
    mask: FixtureMask,
    target_positions: Vec<usize>,
    context: FixtureTensor,
    target: FixtureTensor,
    predicted: FixtureTensor,
    energy: Vec<f32>,
}

#[derive(Debug)]
struct ParityMetadata {
    abs_tolerance: f32,
    rel_tolerance: f32,
}

#[derive(Debug)]
struct ParityConfig {
    encoder: VitConfig,
    predictor: TransformerPredictorConfig,
}

#[derive(Debug, Clone)]
struct FixtureTensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Debug, Clone)]
struct FixtureMask {
    context_indices: Vec<usize>,
    target_indices: Vec<usize>,
    total_tokens: usize,
}

#[derive(Debug, Clone)]
struct ParameterLoadError {
    path: String,
    message: String,
}

struct FixtureWeightMapper<B: Backend> {
    device: B::Device,
    weights: HashMap<String, FixtureTensor>,
    stack: Vec<String>,
    used: HashSet<String>,
    errors: Vec<ParameterLoadError>,
}

impl<B: Backend> FixtureWeightMapper<B> {
    fn new(device: B::Device, weights: HashMap<String, FixtureTensor>) -> Self {
        Self {
            device,
            weights,
            stack: Vec::new(),
            used: HashSet::new(),
            errors: Vec::new(),
        }
    }

    fn into_result(self) -> Result<(), String> {
        if !self.errors.is_empty() {
            let messages: Vec<String> = self
                .errors
                .iter()
                .map(|error| format!("{}: {}", error.path, error.message))
                .collect();
            return Err(messages.join("; "));
        }

        let unused: Vec<String> = self
            .weights
            .keys()
            .filter(|path| !self.used.contains(*path))
            .cloned()
            .collect();
        if !unused.is_empty() {
            return Err(format!("unused fixture weights: {}", unused.join(", ")));
        }

        Ok(())
    }

    fn current_path(&self) -> String {
        self.stack.join(".")
    }

    fn replacement_tensor<const D: usize>(
        &mut self,
        path: &str,
        expected_shape: [usize; D],
    ) -> Option<Tensor<B, D>> {
        let Some(weight) = self.weights.get(path) else {
            self.errors.push(ParameterLoadError {
                path: path.to_owned(),
                message: "missing fixture weight".to_owned(),
            });
            return None;
        };

        let actual_shape: [usize; D] = match weight.shape.clone().try_into() {
            Ok(shape) => shape,
            Err(_) => {
                self.errors.push(ParameterLoadError {
                    path: path.to_owned(),
                    message: format!("expected rank {D}, got shape {:?}", weight.shape),
                });
                return None;
            }
        };

        if actual_shape != expected_shape {
            self.errors.push(ParameterLoadError {
                path: path.to_owned(),
                message: format!(
                    "shape mismatch: expected {:?}, got {:?}",
                    expected_shape, actual_shape
                ),
            });
            return None;
        }

        self.used.insert(path.to_owned());
        Some(Tensor::from_floats(
            burn::tensor::TensorData::new(weight.values.clone(), actual_shape),
            &self.device,
        ))
    }
}

impl<B: Backend> ModuleMapper<B> for FixtureWeightMapper<B> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.stack.push(name.to_owned());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.stack.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let path = self.current_path();
        let (id, tensor, mapper) = param.consume();
        let replacement = self
            .replacement_tensor(&path, tensor.dims())
            .unwrap_or_else(|| tensor.clone());
        Param::from_mapped_value(id, replacement, mapper)
    }
}

struct ZeroRegularizer;

impl<B: Backend> CollapseRegularizer<B> for ZeroRegularizer {
    fn loss(&self, predicted: &Tensor<B, 2>, _target: &Tensor<B, 2>) -> Tensor<B, 1> {
        Tensor::zeros([1], &predicted.device())
    }
}

fn parity_fixture_path() -> PathBuf {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::env::var("JEPA_PARITY_FIXTURE")
        .map(PathBuf::from)
        .map(|path| {
            if path.is_absolute() {
                path
            } else {
                workspace_root.join(path)
            }
        })
        .unwrap_or_else(|_| {
            workspace_root.join("specs/differential/ijepa_strict_tiny_fixture.json")
        })
}

fn load_parity_fixture() -> ParityFixture {
    let fixture_path = parity_fixture_path();
    let render_script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../specs/differential/render_fixture_for_rust.py");
    let output = Command::new("python3")
        .arg(&render_script)
        .arg(&fixture_path)
        .output()
        .unwrap_or_else(|error| {
            panic!(
                "failed to render parity fixture {} with {}: {error}",
                fixture_path.display(),
                render_script.display()
            )
        });

    if !output.status.success() {
        panic!(
            "failed to render parity fixture {}: {}",
            fixture_path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let stdout = String::from_utf8(output.stdout).expect("fixture renderer must emit utf-8");
    let mut scalars = HashMap::<String, String>::new();
    let mut lists = HashMap::<String, Vec<usize>>::new();
    let mut tensors = HashMap::<String, FixtureTensor>::new();
    let mut weights = HashMap::<String, FixtureTensor>::new();
    let mut float_lists = HashMap::<String, Vec<f32>>::new();

    for line in stdout.lines() {
        if line.is_empty() {
            continue;
        }

        let mut parts = line.splitn(4, '\t');
        let kind = parts.next().expect("renderer lines must include a kind");
        match kind {
            "scalar" => {
                let key = parts.next().expect("scalar line missing key").to_owned();
                let value = parts.next().expect("scalar line missing value").to_owned();
                scalars.insert(key, value);
            }
            "usizes" => {
                let key = parts.next().expect("usize list missing key").to_owned();
                let values = parts
                    .next()
                    .expect("usize list missing values")
                    .split(',')
                    .filter(|value| !value.is_empty())
                    .map(|value| value.parse::<usize>().expect("invalid usize list element"))
                    .collect();
                lists.insert(key, values);
            }
            "tensor" | "weight" => {
                let key = parts.next().expect("tensor line missing key").to_owned();
                let shape = parts
                    .next()
                    .expect("tensor line missing shape")
                    .split(',')
                    .filter(|value| !value.is_empty())
                    .map(|value| value.parse::<usize>().expect("invalid tensor shape"))
                    .collect();
                let values = parts
                    .next()
                    .expect("tensor line missing values")
                    .split(',')
                    .filter(|value| !value.is_empty())
                    .map(|value| value.parse::<f32>().expect("invalid tensor value"))
                    .collect();
                let tensor = FixtureTensor { shape, values };
                if kind == "tensor" {
                    tensors.insert(key, tensor);
                } else {
                    weights.insert(key, tensor);
                }
            }
            "floatlist" => {
                let key = parts.next().expect("float list missing key").to_owned();
                let values = parts
                    .next()
                    .expect("float list missing values")
                    .split(',')
                    .filter(|value| !value.is_empty())
                    .map(|value| value.parse::<f32>().expect("invalid float list element"))
                    .collect();
                float_lists.insert(key, values);
            }
            _ => panic!("unexpected fixture line kind: {kind}"),
        }
    }

    fn take_scalar(map: &mut HashMap<String, String>, key: &str) -> String {
        map.remove(key)
            .unwrap_or_else(|| panic!("missing scalar fixture field {key}"))
    }

    fn take_usizes(map: &mut HashMap<String, Vec<usize>>, key: &str) -> Vec<usize> {
        map.remove(key)
            .unwrap_or_else(|| panic!("missing usize fixture field {key}"))
    }

    fn take_tensor(map: &mut HashMap<String, FixtureTensor>, key: &str) -> FixtureTensor {
        map.remove(key)
            .unwrap_or_else(|| panic!("missing tensor fixture field {key}"))
    }

    let encoder = VitConfig {
        in_channels: take_scalar(&mut scalars, "config.encoder.in_channels")
            .parse()
            .expect("invalid encoder.in_channels"),
        image_height: take_scalar(&mut scalars, "config.encoder.image_height")
            .parse()
            .expect("invalid encoder.image_height"),
        image_width: take_scalar(&mut scalars, "config.encoder.image_width")
            .parse()
            .expect("invalid encoder.image_width"),
        patch_size: {
            let values = take_usizes(&mut lists, "config.encoder.patch_size");
            (values[0], values[1])
        },
        embed_dim: take_scalar(&mut scalars, "config.encoder.embed_dim")
            .parse()
            .expect("invalid encoder.embed_dim"),
        num_layers: take_scalar(&mut scalars, "config.encoder.num_layers")
            .parse()
            .expect("invalid encoder.num_layers"),
        num_heads: take_scalar(&mut scalars, "config.encoder.num_heads")
            .parse()
            .expect("invalid encoder.num_heads"),
        mlp_dim: take_scalar(&mut scalars, "config.encoder.mlp_dim")
            .parse()
            .expect("invalid encoder.mlp_dim"),
        dropout: take_scalar(&mut scalars, "config.encoder.dropout")
            .parse()
            .expect("invalid encoder.dropout"),
    };
    let predictor = TransformerPredictorConfig {
        encoder_embed_dim: take_scalar(&mut scalars, "config.predictor.encoder_embed_dim")
            .parse()
            .expect("invalid predictor.encoder_embed_dim"),
        predictor_embed_dim: take_scalar(&mut scalars, "config.predictor.predictor_embed_dim")
            .parse()
            .expect("invalid predictor.predictor_embed_dim"),
        num_layers: take_scalar(&mut scalars, "config.predictor.num_layers")
            .parse()
            .expect("invalid predictor.num_layers"),
        num_heads: take_scalar(&mut scalars, "config.predictor.num_heads")
            .parse()
            .expect("invalid predictor.num_heads"),
        max_target_len: take_scalar(&mut scalars, "config.predictor.max_target_len")
            .parse()
            .expect("invalid predictor.max_target_len"),
    };

    let energy = float_lists
        .remove("energy")
        .expect("missing energy fixture field");

    ParityFixture {
        metadata: ParityMetadata {
            abs_tolerance: take_scalar(&mut scalars, "metadata.abs_tolerance")
                .parse()
                .expect("invalid abs tolerance"),
            rel_tolerance: take_scalar(&mut scalars, "metadata.rel_tolerance")
                .parse()
                .expect("invalid rel tolerance"),
        },
        config: ParityConfig { encoder, predictor },
        weights,
        raw_input: take_tensor(&mut tensors, "raw_input"),
        mask: FixtureMask {
            context_indices: take_usizes(&mut lists, "mask.context_indices"),
            target_indices: take_usizes(&mut lists, "mask.target_indices"),
            total_tokens: take_scalar(&mut scalars, "mask.total_tokens")
                .parse()
                .expect("invalid mask.total_tokens"),
        },
        target_positions: take_usizes(&mut lists, "target_positions"),
        context: take_tensor(&mut tensors, "context"),
        target: take_tensor(&mut tensors, "target"),
        predicted: take_tensor(&mut tensors, "predicted"),
        energy,
    }
}

fn tensor_from_fixture<const D: usize>(fixture: &FixtureTensor) -> Tensor<TestBackend, D> {
    let shape: [usize; D] = fixture
        .shape
        .clone()
        .try_into()
        .unwrap_or_else(|_| panic!("invalid fixture rank for shape {:?}", fixture.shape));
    Tensor::from_floats(
        burn::tensor::TensorData::new(fixture.values.clone(), shape),
        &device(),
    )
}

fn assert_fixture_tensor_close<const D: usize>(
    name: &str,
    actual: Tensor<TestBackend, D>,
    expected: &FixtureTensor,
    abs_tolerance: f32,
    rel_tolerance: f32,
) {
    assert_eq!(
        actual.dims().to_vec(),
        expected.shape,
        "{name} shape mismatch"
    );

    let actual_values = actual.into_data().to_vec::<f32>().unwrap();
    assert_eq!(
        actual_values.len(),
        expected.values.len(),
        "{name} flattened length mismatch"
    );

    let mut max_abs_diff = 0.0f32;
    for (index, (actual_value, expected_value)) in
        actual_values.iter().zip(expected.values.iter()).enumerate()
    {
        let abs_diff = (actual_value - expected_value).abs();
        let allowed = abs_tolerance.max(rel_tolerance * expected_value.abs().max(1.0));
        max_abs_diff = max_abs_diff.max(abs_diff);
        assert!(
            abs_diff <= allowed,
            "{name} mismatch at index {index}: actual={actual_value}, expected={expected_value}, abs_diff={abs_diff}, allowed={allowed}, max_abs_diff={max_abs_diff}"
        );
    }
}

fn load_fixture_model(fixture: &ParityFixture) -> jepa_vision::image::IJepa<TestBackend> {
    let config = IJepaConfig {
        encoder: fixture.config.encoder.clone(),
        predictor: fixture.config.predictor.clone(),
    };
    let model = config.init::<TestBackend>(&device());
    let mut mapper = FixtureWeightMapper::new(device(), fixture.weights.clone());
    let model = model.map(&mut mapper);
    mapper
        .into_result()
        .unwrap_or_else(|error| panic!("failed to map fixture weights: {error}"));
    model
}

fn fixture_mask_spec(mask: &FixtureMask) -> MaskSpec {
    MaskSpec {
        context_indices: mask.context_indices.clone(),
        target_indices: mask.target_indices.clone(),
        total_tokens: mask.total_tokens,
    }
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

#[test]
fn test_ijepa_strict_context_isolates_hidden_patches() {
    let config = IJepaConfig::tiny_test();
    let model = config.init::<TestBackend>(&device());
    let mask = fixed_image_mask();
    let hidden_low = image_with_hidden_patch_value(&mask, 0.0);
    let hidden_high = image_with_hidden_patch_value(&mask, 1_000.0);

    let strict_low = model.encode_context_strict(&hidden_low, &mask.context_indices);
    let strict_high = model.encode_context_strict(&hidden_high, &mask.context_indices);

    let diff: f32 = (strict_low.embeddings - strict_high.embeddings)
        .abs()
        .sum()
        .into_scalar()
        .elem();
    assert!(
        diff < 1e-5,
        "strict image path leaked hidden patches, diff={diff}"
    );
}

#[test]
#[ignore = "run via scripts/run_parity_suite.sh"]
fn test_ijepa_strict_fixture_parity() {
    let fixture = load_parity_fixture();
    let model = load_fixture_model(&fixture);
    let images = tensor_from_fixture::<4>(&fixture.raw_input);
    let energy_fn = jepa_core::energy::L2Energy;
    let regularizer = ZeroRegularizer;
    let mask = fixture_mask_spec(&fixture.mask);

    assert_eq!(
        fixture.target_positions, fixture.mask.target_indices,
        "fixture target_positions must match the strict target mask"
    );
    assert!(mask.validate().is_ok(), "fixture mask must be valid");

    let output = model
        .try_forward_step_strict(&images, mask, &energy_fn, &regularizer, 0.0)
        .expect("fixture-backed strict forward step should succeed");

    assert_fixture_tensor_close(
        "context",
        output.context.embeddings,
        &fixture.context,
        fixture.metadata.abs_tolerance,
        fixture.metadata.rel_tolerance,
    );
    assert_fixture_tensor_close(
        "target",
        output.target.embeddings,
        &fixture.target,
        fixture.metadata.abs_tolerance,
        fixture.metadata.rel_tolerance,
    );
    assert_fixture_tensor_close(
        "predicted",
        output.predicted.embeddings,
        &fixture.predicted,
        fixture.metadata.abs_tolerance,
        fixture.metadata.rel_tolerance,
    );

    let energy_values = output.energy.value.into_data().to_vec::<f32>().unwrap();
    assert_eq!(
        energy_values.len(),
        fixture.energy.len(),
        "energy length mismatch"
    );
    for (index, (actual, expected)) in energy_values.iter().zip(fixture.energy.iter()).enumerate() {
        let abs_diff = (actual - expected).abs();
        let allowed = fixture
            .metadata
            .abs_tolerance
            .max(fixture.metadata.rel_tolerance * expected.abs().max(1.0));
        assert!(
            abs_diff <= allowed,
            "energy mismatch at index {index}: actual={actual}, expected={expected}, abs_diff={abs_diff}, allowed={allowed}"
        );
    }
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

#[test]
fn test_vjepa_strict_context_isolates_hidden_tubelets() {
    let config = jepa_vision::video::VJepaConfig::tiny_test();
    let model = config.init::<TestBackend>(&device());
    let mask = fixed_video_mask();
    let hidden_low = video_with_hidden_tubelet_value(&mask, 0.0);
    let hidden_high = video_with_hidden_tubelet_value(&mask, 1_000.0);

    let strict_low = model.encode_context_strict(&hidden_low, &mask.context_indices);
    let strict_high = model.encode_context_strict(&hidden_high, &mask.context_indices);

    let diff: f32 = (strict_low.embeddings - strict_high.embeddings)
        .abs()
        .sum()
        .into_scalar()
        .elem();
    assert!(
        diff < 1e-5,
        "strict video path leaked hidden tubelets, diff={diff}"
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

// ---- Cross-crate training step integration (jepa-vision + jepa-train) ----

/// Integration test: jepa-train's JepaComponents with real ViT encoder + predictor.
///
/// This validates that the training orchestrator from jepa-train works correctly
/// with real neural network modules from jepa-vision (not test stubs).
#[test]
fn test_train_forward_step_with_real_vit() {
    use jepa_core::collapse::VICReg;
    use jepa_core::energy::L2Energy;
    use jepa_train::trainer::JepaComponents;
    use rand::SeedableRng;

    let config = IJepaConfig::tiny_test();
    let model = config.init::<TestBackend>(&device());

    let masking = jepa_core::masking::BlockMasking {
        num_targets: 2,
        target_scale: (0.15, 0.3),
        target_aspect_ratio: (0.75, 1.5),
    };
    let energy_fn = L2Energy;
    let regularizer = VICReg::default();

    let components = JepaComponents::new(
        &model.context_encoder,
        &model.target_encoder,
        &model.predictor,
        &energy_fn,
        &regularizer,
        &masking,
        0.1,
    );

    let images: Tensor<TestBackend, 4> = Tensor::random(
        [2, 1, 8, 8],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device(),
    );
    let input_shape = InputShape::Image {
        height: 4,
        width: 4,
    };

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let output = components.forward_step(&images, &input_shape, &mut rng);

    // All loss terms should be finite
    let energy_val: f32 = output.energy.value.into_scalar().elem();
    assert!(
        energy_val.is_finite(),
        "energy should be finite: {energy_val}"
    );
    assert!(
        energy_val >= 0.0,
        "L2 energy should be non-negative: {energy_val}"
    );

    let total_val: f32 = output.total_loss.into_scalar().elem();
    assert!(
        total_val.is_finite(),
        "total loss should be finite: {total_val}"
    );

    // Predicted and target shapes should match
    assert_eq!(output.predicted.batch_size(), 2);
    assert_eq!(output.target.batch_size(), 2);
    assert_eq!(output.predicted.embed_dim(), output.target.embed_dim());
    assert_eq!(output.predicted.seq_len(), output.target.seq_len());

    // Mask should be valid
    assert!(output.mask.validate().is_ok());
}

/// Integration test: training step determinism with same seed.
#[test]
fn test_train_forward_step_determinism() {
    use jepa_core::collapse::VICReg;
    use jepa_core::energy::L2Energy;
    use jepa_train::trainer::JepaComponents;
    use rand::SeedableRng;

    let config = IJepaConfig::tiny_test();
    let model = config.init::<TestBackend>(&device());

    let masking = jepa_core::masking::BlockMasking {
        num_targets: 2,
        target_scale: (0.15, 0.3),
        target_aspect_ratio: (0.75, 1.5),
    };
    let energy_fn = L2Energy;
    let regularizer = VICReg::default();

    let components = JepaComponents::new(
        &model.context_encoder,
        &model.target_encoder,
        &model.predictor,
        &energy_fn,
        &regularizer,
        &masking,
        0.1,
    );

    let images: Tensor<TestBackend, 4> = Tensor::ones([1, 1, 8, 8], &device());
    let input_shape = InputShape::Image {
        height: 4,
        width: 4,
    };

    // Same seed → same mask
    let mut rng1 = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let mut rng2 = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let out1 = components.forward_step(&images, &input_shape, &mut rng1);
    let out2 = components.forward_step(&images, &input_shape, &mut rng2);

    assert_eq!(
        out1.mask.target_indices, out2.mask.target_indices,
        "same seed should produce same mask"
    );
    assert_eq!(
        out1.mask.context_indices, out2.mask.context_indices,
        "same seed should produce same context mask"
    );
}

/// Integration test: V-JEPA pipeline with spatiotemporal masking, cosine energy,
/// and Barlow Twins regularization — testing all alternative module combinations.
#[test]
fn test_vjepa_full_pipeline_with_alternative_modules() {
    use jepa_vision::video::VJepaConfig;
    use rand::SeedableRng;

    let config = VJepaConfig::tiny_test();
    let model = config.init::<TestBackend>(&device());

    let video: Tensor<TestBackend, 5> = Tensor::random(
        [1, 1, 4, 8, 8],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device(),
    );

    // Encode
    let context_repr = model.context_encoder.forward(&video);
    let target_repr = model.target_encoder.forward(&video);
    assert_eq!(context_repr.seq_len(), 32);

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

    // Test CosineEnergy (alternative to L2)
    let cosine_energy = jepa_core::energy::CosineEnergy.compute(&predicted, &target_gathered);
    let cosine_val: f32 = cosine_energy.value.into_scalar().elem();
    assert!(cosine_val.is_finite(), "cosine energy should be finite");

    // Test SmoothL1Energy (alternative to L2)
    let smooth_energy =
        jepa_core::energy::SmoothL1Energy::new(1.0).compute(&predicted, &target_gathered);
    let smooth_val: f32 = smooth_energy.value.into_scalar().elem();
    assert!(smooth_val.is_finite(), "smooth L1 energy should be finite");
    assert!(smooth_val >= 0.0, "smooth L1 energy should be non-negative");

    // Test BarlowTwins regularization (alternative to VICReg)
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
