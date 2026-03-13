//! Load model weights from SafeTensors files.
//!
//! [SafeTensors](https://huggingface.co/docs/safetensors) is the recommended
//! serialization format for PyTorch / HuggingFace model checkpoints. This
//! module reads `.safetensors` files, applies optional key remapping (via
//! [`crate::keymap`]), and returns burn-compatible tensors ready for loading
//! into jepa-rs models.
//!
//! ## Supported dtypes
//!
//! | SafeTensors dtype | Conversion |
//! |-------------------|------------|
//! | `F32` | Direct mapping |
//! | `F16` | Widened to f32 via `half` crate |
//! | `BF16` | Widened to f32 via `half` crate |

use std::collections::HashMap;
use std::path::Path;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::keymap::KeyMapping;

/// Errors from safetensors loading.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// Failed to read the safetensors file from disk.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to parse the safetensors format.
    #[error("safetensors parse error: {0}")]
    Parse(String),

    /// Unsupported tensor dtype in the checkpoint.
    #[error("unsupported dtype: {0:?}")]
    UnsupportedDtype(safetensors::Dtype),

    /// A required key was not found in the checkpoint.
    #[error("missing key in checkpoint: {0}")]
    MissingKey(String),

    /// Shape mismatch between checkpoint tensor and model parameter.
    #[error("shape mismatch for {key}: checkpoint {checkpoint_shape:?} vs model {model_shape:?}")]
    ShapeMismatch {
        key: String,
        checkpoint_shape: Vec<usize>,
        model_shape: Vec<usize>,
    },
}

/// A loaded tensor with its metadata.
///
/// # Example
///
/// ```
/// use jepa_compat::safetensors::LoadedTensor;
/// use burn_ndarray::NdArray;
///
/// type B = NdArray<f32>;
/// let device = burn_ndarray::NdArrayDevice::Cpu;
///
/// let tensor = LoadedTensor {
///     data: vec![1.0, 2.0, 3.0, 4.0],
///     shape: vec![2, 2],
///     original_key: "module.weight".to_string(),
///     burn_key: "weight".to_string(),
/// };
/// let burn_tensor: burn::tensor::Tensor<B, 2> = tensor.to_tensor(&device);
/// assert_eq!(burn_tensor.dims(), [2, 2]);
/// ```
#[derive(Debug, Clone)]
pub struct LoadedTensor {
    /// The tensor data as f32 values.
    pub data: Vec<f32>,
    /// The shape of the tensor.
    pub shape: Vec<usize>,
    /// The original key in the safetensors file.
    pub original_key: String,
    /// The mapped burn parameter name.
    pub burn_key: String,
}

impl LoadedTensor {
    /// Convert to a burn TensorData.
    pub fn to_tensor_data(&self) -> TensorData {
        TensorData::new(self.data.clone(), self.shape.clone())
    }

    /// Convert to a burn Tensor on the given device.
    ///
    /// # Type Parameters
    /// * `B` - The burn backend
    /// * `D` - The number of dimensions
    pub fn to_tensor<B: Backend, const D: usize>(&self, device: &B::Device) -> Tensor<B, D> {
        Tensor::from_data(self.to_tensor_data(), device)
    }
}

/// A collection of loaded tensors from a checkpoint.
#[derive(Debug)]
pub struct Checkpoint {
    /// All loaded tensors, keyed by their burn parameter name.
    pub tensors: HashMap<String, LoadedTensor>,
    /// Keys from the checkpoint that were not mapped.
    pub unmapped_keys: Vec<String>,
}

impl Checkpoint {
    /// Get a tensor by its burn parameter name.
    pub fn get(&self, burn_key: &str) -> Option<&LoadedTensor> {
        self.tensors.get(burn_key)
    }

    /// Get a tensor as a burn Tensor, or return an error if missing.
    pub fn get_tensor<B: Backend, const D: usize>(
        &self,
        burn_key: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, D>, LoadError> {
        self.tensors
            .get(burn_key)
            .map(|t| t.to_tensor::<B, D>(device))
            .ok_or_else(|| LoadError::MissingKey(burn_key.to_string()))
    }

    /// Validate that all expected keys are present and shapes match.
    pub fn validate_shapes(&self, expected: &HashMap<String, Vec<usize>>) -> Result<(), LoadError> {
        for (key, expected_shape) in expected {
            match self.tensors.get(key) {
                None => return Err(LoadError::MissingKey(key.clone())),
                Some(tensor) => {
                    if tensor.shape != *expected_shape {
                        return Err(LoadError::ShapeMismatch {
                            key: key.clone(),
                            checkpoint_shape: tensor.shape.clone(),
                            model_shape: expected_shape.clone(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Get all burn parameter names.
    pub fn keys(&self) -> Vec<&str> {
        self.tensors.keys().map(|k| k.as_str()).collect()
    }

    /// Number of loaded tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether no tensors were loaded.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Load a safetensors checkpoint from a file path.
///
/// Reads the safetensors file, converts all tensors to f32, and maps
/// keys according to the provided key mappings.
///
/// # Arguments
/// * `path` - Path to the `.safetensors` file
/// * `mappings` - Key mapping rules from [`crate::keymap`]
///
/// # Returns
/// A [`Checkpoint`] containing all mapped tensors and a list of unmapped keys.
pub fn load_checkpoint(
    path: impl AsRef<Path>,
    mappings: &[KeyMapping],
) -> Result<Checkpoint, LoadError> {
    let data = std::fs::read(path)?;
    load_checkpoint_from_bytes(&data, mappings)
}

/// Load a safetensors checkpoint from raw bytes.
///
/// This is the core loading function. It parses the safetensors format,
/// converts tensors to f32, and applies key mappings.
pub fn load_checkpoint_from_bytes(
    data: &[u8],
    mappings: &[KeyMapping],
) -> Result<Checkpoint, LoadError> {
    let st =
        safetensors::SafeTensors::deserialize(data).map_err(|e| LoadError::Parse(e.to_string()))?;

    let mut tensors = HashMap::new();
    let mut unmapped_keys = Vec::new();

    for (key, view) in st.tensors() {
        let stripped = crate::keymap::strip_prefix(&key);
        let burn_key = match crate::keymap::resolve_key(stripped, mappings) {
            Some(k) => k,
            None => {
                unmapped_keys.push(key.to_string());
                continue;
            }
        };

        let shape: Vec<usize> = view.shape().to_vec();
        let f32_data = convert_to_f32(view.dtype(), view.data())?;

        tensors.insert(
            burn_key.clone(),
            LoadedTensor {
                data: f32_data,
                shape,
                original_key: key.to_string(),
                burn_key,
            },
        );
    }

    Ok(Checkpoint {
        tensors,
        unmapped_keys,
    })
}

/// Convert raw tensor bytes to f32 values based on dtype.
fn convert_to_f32(dtype: safetensors::Dtype, data: &[u8]) -> Result<Vec<f32>, LoadError> {
    match dtype {
        safetensors::Dtype::F32 => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(floats)
        }
        safetensors::Dtype::F16 => {
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect();
            Ok(floats)
        }
        safetensors::Dtype::BF16 => {
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect();
            Ok(floats)
        }
        other => Err(LoadError::UnsupportedDtype(other)),
    }
}

/// Create a minimal safetensors file in memory for testing.
///
/// This is useful for unit tests that need to verify the loading pipeline
/// without requiring actual model checkpoint files.
#[cfg(test)]
fn create_test_safetensors(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
    use std::collections::BTreeMap;

    // Build the metadata header
    let mut header = BTreeMap::new();
    let mut data_buf = Vec::new();

    for &(name, shape, values) in tensors {
        let start = data_buf.len();
        for &v in values {
            data_buf.extend_from_slice(&v.to_le_bytes());
        }
        let end = data_buf.len();

        header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }

    let header_json = serde_json::to_string(&header).unwrap();
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut out = Vec::new();
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&data_buf);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keymap::ijepa_vit_keymap;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

    #[test]
    fn test_load_simple_checkpoint() {
        let data = create_test_safetensors(&[
            ("norm.weight", &[16], &[1.0f32; 16]),
            ("norm.bias", &[16], &[0.0f32; 16]),
        ]);

        let mappings = ijepa_vit_keymap();
        let ckpt = load_checkpoint_from_bytes(&data, &mappings).unwrap();

        assert_eq!(ckpt.len(), 2);
        assert!(ckpt.get("norm.weight").is_some());
        assert!(ckpt.get("norm.bias").is_some());
    }

    #[test]
    fn test_load_with_prefix_stripping() {
        let data = create_test_safetensors(&[("module.norm.weight", &[16], &[1.0f32; 16])]);

        let mappings = ijepa_vit_keymap();
        let ckpt = load_checkpoint_from_bytes(&data, &mappings).unwrap();

        assert!(ckpt.get("norm.weight").is_some());
    }

    #[test]
    fn test_load_layer_keys() {
        let data = create_test_safetensors(&[
            ("blocks.0.norm1.weight", &[32], &[1.0f32; 32]),
            ("blocks.0.attn.proj.weight", &[32, 32], &[0.5f32; 32 * 32]),
        ]);

        let mappings = ijepa_vit_keymap();
        let ckpt = load_checkpoint_from_bytes(&data, &mappings).unwrap();

        assert!(ckpt.get("blocks.0.norm1.weight").is_some());
        // attn.proj → attn.out_proj
        assert!(ckpt.get("blocks.0.attn.out_proj.weight").is_some());
    }

    #[test]
    fn test_unmapped_keys_tracked() {
        let data = create_test_safetensors(&[
            ("norm.weight", &[16], &[1.0f32; 16]),
            ("some.unknown.key", &[8], &[0.0f32; 8]),
        ]);

        let mappings = ijepa_vit_keymap();
        let ckpt = load_checkpoint_from_bytes(&data, &mappings).unwrap();

        assert_eq!(ckpt.len(), 1);
        assert_eq!(ckpt.unmapped_keys, vec!["some.unknown.key"]);
    }

    #[test]
    fn test_tensor_shape_preserved() {
        let shape = &[4, 8];
        let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let data = create_test_safetensors(&[("norm.weight", shape, &values)]);

        let mappings = ijepa_vit_keymap();
        let ckpt = load_checkpoint_from_bytes(&data, &mappings).unwrap();

        let tensor = ckpt.get("norm.weight").unwrap();
        assert_eq!(tensor.shape, vec![4, 8]);
        assert_eq!(tensor.data.len(), 32);
        assert_eq!(tensor.data[0], 0.0);
        assert_eq!(tensor.data[31], 31.0);
    }

    #[test]
    fn test_to_burn_tensor() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let data = create_test_safetensors(&[("norm.weight", &[4], &values)]);

        let mappings = ijepa_vit_keymap();
        let ckpt = load_checkpoint_from_bytes(&data, &mappings).unwrap();

        let tensor: Tensor<TestBackend, 1> = ckpt
            .get_tensor::<TestBackend, 1>("norm.weight", &device())
            .unwrap();
        assert_eq!(tensor.dims(), [4]);
    }

    #[test]
    fn test_validate_shapes_ok() {
        let data = create_test_safetensors(&[
            ("norm.weight", &[16], &[1.0f32; 16]),
            ("norm.bias", &[16], &[0.0f32; 16]),
        ]);

        let mappings = ijepa_vit_keymap();
        let ckpt = load_checkpoint_from_bytes(&data, &mappings).unwrap();

        let mut expected = HashMap::new();
        expected.insert("norm.weight".to_string(), vec![16]);
        expected.insert("norm.bias".to_string(), vec![16]);

        assert!(ckpt.validate_shapes(&expected).is_ok());
    }

    #[test]
    fn test_validate_shapes_mismatch() {
        let data = create_test_safetensors(&[("norm.weight", &[16], &[1.0f32; 16])]);

        let mappings = ijepa_vit_keymap();
        let ckpt = load_checkpoint_from_bytes(&data, &mappings).unwrap();

        let mut expected = HashMap::new();
        expected.insert("norm.weight".to_string(), vec![32]);

        assert!(matches!(
            ckpt.validate_shapes(&expected),
            Err(LoadError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_validate_shapes_missing() {
        let data = create_test_safetensors(&[("norm.weight", &[16], &[1.0f32; 16])]);

        let mappings = ijepa_vit_keymap();
        let ckpt = load_checkpoint_from_bytes(&data, &mappings).unwrap();

        let mut expected = HashMap::new();
        expected.insert("norm.bias".to_string(), vec![16]);

        assert!(matches!(
            ckpt.validate_shapes(&expected),
            Err(LoadError::MissingKey(_))
        ));
    }

    #[test]
    fn test_checkpoint_is_empty() {
        let data = create_test_safetensors(&[("unknown.key", &[4], &[0.0f32; 4])]);

        let mappings = ijepa_vit_keymap();
        let ckpt = load_checkpoint_from_bytes(&data, &mappings).unwrap();
        assert!(ckpt.is_empty());
    }

    #[test]
    fn test_f32_conversion() {
        let values = vec![1.5f32, -2.5, 3.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = convert_to_f32(safetensors::Dtype::F32, &bytes).unwrap();
        assert_eq!(result, values);
    }

    #[test]
    fn test_f16_conversion() {
        let f16_val = half::f16::from_f32(1.5);
        let bytes = f16_val.to_bits().to_le_bytes();
        let result = convert_to_f32(safetensors::Dtype::F16, &bytes).unwrap();
        assert!((result[0] - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_bf16_conversion() {
        let bf16_val = half::bf16::from_f32(2.0);
        let bytes = bf16_val.to_bits().to_le_bytes();
        let result = convert_to_f32(safetensors::Dtype::BF16, &bytes).unwrap();
        assert!((result[0] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_loaded_tensor_original_key_preserved() {
        let data = create_test_safetensors(&[(
            "module.blocks.7.attn.proj.weight",
            &[4, 4],
            &[1.0f32; 16],
        )]);

        let mappings = ijepa_vit_keymap();
        let ckpt = load_checkpoint_from_bytes(&data, &mappings).unwrap();

        let tensor = ckpt.get("blocks.7.attn.out_proj.weight").unwrap();
        assert_eq!(tensor.original_key, "module.blocks.7.attn.proj.weight");
        assert_eq!(tensor.burn_key, "blocks.7.attn.out_proj.weight");
    }
}
