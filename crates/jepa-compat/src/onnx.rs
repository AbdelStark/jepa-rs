//! ONNX model inspection and initializer loading for JEPA models.
//!
//! This module uses the `tract-onnx` protobuf parser to extract
//! metadata and weight initializers from `.onnx` files **without**
//! executing the computation graph. For graph execution, see the
//! [`crate::runtime`] module.
//!
//! Two primary use cases:
//!
//! 1. **Inspection** — [`OnnxModelInfo::from_file`] returns input/output
//!    shapes, data types, and tensor metadata for a model file.
//! 2. **Weight import** — [`load_checkpoint`] extracts all initializer
//!    tensors into a [`Checkpoint`]
//!    compatible with the key-remapping infrastructure.
//!
//! ## Usage
//!
//! ```no_run
//! use jepa_compat::onnx::{load_checkpoint, OnnxKeyMap, OnnxModelInfo};
//!
//! let info = OnnxModelInfo::from_file("model.onnx")?;
//! println!("Inputs: {:?}", info.inputs);
//!
//! let checkpoint = load_checkpoint("model.onnx", &OnnxKeyMap::new())?;
//! println!("Loaded {} tensors", checkpoint.len());
//! # Ok::<(), jepa_compat::onnx::OnnxError>(())
//! ```

use std::collections::HashMap;
use std::path::Path;

use prost::Message;
use tract_onnx::data_resolver::MmapDataResolver;
use tract_onnx::pb::{
    tensor_proto, tensor_shape_proto, type_proto, ModelProto, TensorProto, TensorShapeProto,
    ValueInfoProto,
};

use crate::safetensors::{Checkpoint, LoadedTensor};

/// Metadata about an ONNX model.
///
/// Contains information extracted from the ONNX model graph
/// without loading the full weights into burn tensors.
#[derive(Debug, Clone)]
pub struct OnnxModelInfo {
    /// Model name from the ONNX graph.
    pub name: String,
    /// Producer that created the ONNX model (e.g., "pytorch").
    pub producer: String,
    /// ONNX opset version.
    pub opset_version: i64,
    /// Input tensor specifications.
    pub inputs: Vec<OnnxTensorInfo>,
    /// Output tensor specifications.
    pub outputs: Vec<OnnxTensorInfo>,
}

/// Information about a single ONNX tensor (input or output).
#[derive(Debug, Clone)]
pub struct OnnxTensorInfo {
    /// Tensor name in the ONNX graph.
    pub name: String,
    /// Shape dimensions. Dynamic dimensions are represented as -1.
    pub shape: Vec<i64>,
    /// Element data type.
    pub dtype: OnnxDtype,
}

/// ONNX data types relevant for JEPA models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxDtype {
    /// 32-bit floating point.
    Float32,
    /// 16-bit floating point.
    Float16,
    /// Brain floating point (bfloat16).
    BFloat16,
    /// 64-bit floating point.
    Float64,
    /// Unknown or unsupported data type.
    Unknown,
}

impl OnnxDtype {
    fn from_elem_type(elem_type: i32) -> Self {
        match tensor_proto::DataType::from_i32(elem_type) {
            Some(tensor_proto::DataType::Float) => Self::Float32,
            Some(tensor_proto::DataType::Float16) => Self::Float16,
            Some(tensor_proto::DataType::Bfloat16) => Self::BFloat16,
            Some(tensor_proto::DataType::Double) => Self::Float64,
            _ => Self::Unknown,
        }
    }
}

/// Errors from ONNX loading operations.
#[derive(Debug, thiserror::Error)]
pub enum OnnxError {
    /// The ONNX file could not be read.
    #[error("failed to read ONNX file: {path}")]
    FileNotFound { path: String },
    /// The ONNX model format is invalid or unsupported.
    #[error("invalid ONNX model: {reason}")]
    InvalidModel { reason: String },
    /// A required opset version is not supported.
    #[error("unsupported opset version: {version}")]
    UnsupportedOpset { version: i64 },
    /// Weight shape mismatch between ONNX model and target burn model.
    #[error("shape mismatch for tensor '{name}': expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<i64>,
    },
    /// Tensor dtype is not currently supported for loading.
    #[error("unsupported ONNX tensor dtype for '{name}': {dtype:?}")]
    UnsupportedDtype { name: String, dtype: OnnxDtype },
    /// Reserved for future builds that may gate an ONNX runtime surface.
    #[error("ONNX runtime not available")]
    RuntimeNotAvailable,
}

/// Mapping from ONNX weight names to burn module parameter paths.
///
/// Similar to [`crate::keymap`] for safetensors, but handles the
/// naming conventions used by ONNX exports.
#[derive(Debug, Clone, Default)]
pub struct OnnxKeyMap {
    /// Explicit key remappings: ONNX name -> burn path.
    pub remappings: HashMap<String, String>,
    /// Prefix to strip from ONNX tensor names.
    pub strip_prefix: Option<String>,
}

impl OnnxKeyMap {
    /// Create a new empty key map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a prefix to strip from all ONNX tensor names.
    pub fn with_strip_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.strip_prefix = Some(prefix.into());
        self
    }

    /// Add an explicit remapping.
    pub fn with_remap(mut self, onnx_key: impl Into<String>, burn_key: impl Into<String>) -> Self {
        self.remappings.insert(onnx_key.into(), burn_key.into());
        self
    }

    /// Map an ONNX tensor name to a burn parameter path.
    pub fn map_key(&self, onnx_key: &str) -> Option<String> {
        if let Some(mapped) = self.remappings.get(onnx_key) {
            return Some(mapped.clone());
        }

        let key = match &self.strip_prefix {
            Some(prefix) => onnx_key.strip_prefix(prefix.as_str()).unwrap_or(onnx_key),
            None => onnx_key,
        };

        Some(key.to_string())
    }
}

impl OnnxModelInfo {
    /// Load model info from an ONNX file path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, OnnxError> {
        let proto = decode_model_from_path(path.as_ref())?;
        from_model_proto(&proto)
    }

    /// Load model info from raw ONNX bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, OnnxError> {
        let proto = decode_model(data)?;
        from_model_proto(&proto)
    }
}

/// Load ONNX initializers into the same checkpoint structure used for safetensors.
pub fn load_checkpoint(
    path: impl AsRef<Path>,
    key_map: &OnnxKeyMap,
) -> Result<Checkpoint, OnnxError> {
    let path = path.as_ref();
    let proto = decode_model_from_path(path)?;
    load_checkpoint_from_proto(&proto, key_map, path.parent())
}

/// Load ONNX initializers from raw model bytes.
pub fn load_checkpoint_from_bytes(
    data: &[u8],
    key_map: &OnnxKeyMap,
) -> Result<Checkpoint, OnnxError> {
    let proto = decode_model(data)?;
    load_checkpoint_from_proto(&proto, key_map, None)
}

fn decode_model_from_path(path: &Path) -> Result<ModelProto, OnnxError> {
    if !path.exists() {
        return Err(OnnxError::FileNotFound {
            path: path.display().to_string(),
        });
    }

    let data = std::fs::read(path).map_err(|_| OnnxError::FileNotFound {
        path: path.display().to_string(),
    })?;
    decode_model(&data)
}

fn decode_model(data: &[u8]) -> Result<ModelProto, OnnxError> {
    ModelProto::decode(data).map_err(|err| OnnxError::InvalidModel {
        reason: err.to_string(),
    })
}

fn from_model_proto(proto: &ModelProto) -> Result<OnnxModelInfo, OnnxError> {
    let graph = proto
        .graph
        .as_ref()
        .ok_or_else(|| OnnxError::InvalidModel {
            reason: "model does not contain a graph".to_string(),
        })?;
    let opset_version = opset_version(proto)?;

    Ok(OnnxModelInfo {
        name: graph.name.clone(),
        producer: proto.producer_name.clone(),
        opset_version,
        inputs: graph
            .input
            .iter()
            .map(value_info_to_tensor_info)
            .collect::<Result<Vec<_>, _>>()?,
        outputs: graph
            .output
            .iter()
            .map(value_info_to_tensor_info)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn opset_version(proto: &ModelProto) -> Result<i64, OnnxError> {
    let version = proto
        .opset_import
        .iter()
        .find(|import| import.domain.is_empty() || import.domain == "ai.onnx")
        .map(|import| import.version)
        .unwrap_or(0);

    if version <= 0 {
        return Err(OnnxError::InvalidModel {
            reason: "model is missing a usable ai.onnx opset import".to_string(),
        });
    }

    Ok(version)
}

fn value_info_to_tensor_info(info: &ValueInfoProto) -> Result<OnnxTensorInfo, OnnxError> {
    let tensor_type = match info.r#type.as_ref().and_then(|ty| ty.value.as_ref()) {
        Some(type_proto::Value::TensorType(tensor)) => tensor,
        None => {
            return Err(OnnxError::InvalidModel {
                reason: format!(
                    "value '{}' does not carry tensor type information",
                    info.name
                ),
            });
        }
    };

    Ok(OnnxTensorInfo {
        name: info.name.clone(),
        shape: tensor_shape(tensor_type.shape.as_ref()),
        dtype: OnnxDtype::from_elem_type(tensor_type.elem_type),
    })
}

fn tensor_shape(shape: Option<&TensorShapeProto>) -> Vec<i64> {
    shape
        .map(|shape| {
            shape
                .dim
                .iter()
                .map(|dim| match dim.value.as_ref() {
                    Some(tensor_shape_proto::dimension::Value::DimValue(value)) => *value,
                    Some(tensor_shape_proto::dimension::Value::DimParam(_)) | None => -1,
                })
                .collect()
        })
        .unwrap_or_default()
}

fn load_checkpoint_from_proto(
    proto: &ModelProto,
    key_map: &OnnxKeyMap,
    model_dir: Option<&Path>,
) -> Result<Checkpoint, OnnxError> {
    let graph = proto
        .graph
        .as_ref()
        .ok_or_else(|| OnnxError::InvalidModel {
            reason: "model does not contain a graph".to_string(),
        })?;

    let mut tensors = HashMap::new();
    let mut unmapped_keys = Vec::new();
    let model_dir = model_dir.map(|dir| dir.to_string_lossy().into_owned());

    for initializer in &graph.initializer {
        let original_key = initializer.name.clone();
        let Some(burn_key) = key_map.map_key(&original_key) else {
            unmapped_keys.push(original_key);
            continue;
        };

        let loaded =
            initializer_to_loaded_tensor(initializer, burn_key.clone(), model_dir.as_deref())?;
        tensors.insert(burn_key, loaded);
    }

    Ok(Checkpoint {
        tensors,
        unmapped_keys,
    })
}

fn initializer_to_loaded_tensor(
    tensor: &TensorProto,
    burn_key: String,
    model_dir: Option<&str>,
) -> Result<LoadedTensor, OnnxError> {
    let dtype = OnnxDtype::from_elem_type(tensor.data_type);
    if dtype == OnnxDtype::Unknown {
        return Err(OnnxError::UnsupportedDtype {
            name: tensor.name.clone(),
            dtype,
        });
    }

    let loaded =
        tract_onnx::tensor::load_tensor(&MmapDataResolver, tensor, model_dir).map_err(|err| {
            OnnxError::InvalidModel {
                reason: format!("failed to load initializer '{}': {err}", tensor.name),
            }
        })?;
    let loaded = loaded
        .cast_to::<f32>()
        .map_err(|err| OnnxError::InvalidModel {
            reason: format!("failed to cast initializer '{}' to f32: {err}", tensor.name),
        })?;

    Ok(LoadedTensor {
        data: loaded
            .as_slice::<f32>()
            .map_err(|err| OnnxError::InvalidModel {
                reason: format!(
                    "failed to materialize initializer '{}' as f32 slice: {err}",
                    tensor.name
                ),
            })?
            .to_vec(),
        shape: loaded.shape().to_vec(),
        original_key: tensor.name.clone(),
        burn_key,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_onnx::pb::{GraphProto, OperatorSetIdProto, TypeProto};

    fn tensor_type(shape: &[DimensionSpec], elem_type: tensor_proto::DataType) -> TypeProto {
        TypeProto {
            denotation: String::new(),
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: elem_type as i32,
                shape: Some(TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|dimension| tensor_shape_proto::Dimension {
                            denotation: String::new(),
                            value: Some(match dimension {
                                DimensionSpec::Value(value) => {
                                    tensor_shape_proto::dimension::Value::DimValue(*value)
                                }
                                DimensionSpec::Param(name) => {
                                    tensor_shape_proto::dimension::Value::DimParam(
                                        (*name).to_string(),
                                    )
                                }
                            }),
                        })
                        .collect(),
                }),
            })),
        }
    }

    fn value_info(
        name: &str,
        shape: &[DimensionSpec],
        elem_type: tensor_proto::DataType,
    ) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(tensor_type(shape, elem_type)),
            doc_string: String::new(),
        }
    }

    fn float_tensor(name: &str, shape: &[i64], values: &[f32]) -> TensorProto {
        TensorProto {
            dims: shape.to_vec(),
            data_type: tensor_proto::DataType::Float as i32,
            segment: None,
            float_data: values.to_vec(),
            int32_data: Vec::new(),
            string_data: Vec::new(),
            int64_data: Vec::new(),
            name: name.to_string(),
            doc_string: String::new(),
            raw_data: Vec::new(),
            double_data: Vec::new(),
            uint64_data: Vec::new(),
            data_location: None,
            external_data: Vec::new(),
        }
    }

    fn minimal_model_bytes() -> Vec<u8> {
        let model = ModelProto {
            ir_version: 9,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            producer_name: "pytorch".to_string(),
            producer_version: "2.3".to_string(),
            domain: String::new(),
            model_version: 1,
            doc_string: String::new(),
            graph: Some(GraphProto {
                node: Vec::new(),
                name: "tiny-jepa".to_string(),
                initializer: vec![float_tensor(
                    "model.encoder.weight",
                    &[2, 2],
                    &[1.0, 2.0, 3.0, 4.0],
                )],
                sparse_initializer: Vec::new(),
                doc_string: String::new(),
                input: vec![value_info(
                    "input",
                    &[
                        DimensionSpec::Value(1),
                        DimensionSpec::Value(3),
                        DimensionSpec::Value(224),
                        DimensionSpec::Value(224),
                    ],
                    tensor_proto::DataType::Float,
                )],
                output: vec![value_info(
                    "output",
                    &[
                        DimensionSpec::Value(1),
                        DimensionSpec::Param("tokens"),
                        DimensionSpec::Value(768),
                    ],
                    tensor_proto::DataType::Float,
                )],
                value_info: Vec::new(),
                quantization_annotation: Vec::new(),
            }),
            metadata_props: Vec::new(),
            training_info: Vec::new(),
            functions: Vec::new(),
        };

        model.encode_to_vec()
    }

    #[derive(Clone, Copy)]
    enum DimensionSpec<'a> {
        Value(i64),
        Param(&'a str),
    }

    #[test]
    fn test_missing_onnx_file_returns_file_not_found() {
        let result = OnnxModelInfo::from_file("nonexistent.onnx");
        assert!(matches!(result, Err(OnnxError::FileNotFound { .. })));
    }

    #[test]
    fn test_model_info_from_real_onnx_bytes() {
        let info = OnnxModelInfo::from_bytes(&minimal_model_bytes()).unwrap();

        assert_eq!(info.name, "tiny-jepa");
        assert_eq!(info.producer, "pytorch");
        assert_eq!(info.opset_version, 17);
        assert_eq!(info.inputs.len(), 1);
        assert_eq!(info.outputs.len(), 1);
        assert_eq!(info.inputs[0].shape, vec![1, 3, 224, 224]);
        assert_eq!(info.outputs[0].shape, vec![1, -1, 768]);
        assert_eq!(info.outputs[0].dtype, OnnxDtype::Float32);
    }

    #[test]
    fn test_load_checkpoint_from_onnx_bytes() {
        let key_map = OnnxKeyMap::new().with_strip_prefix("model.");
        let checkpoint = load_checkpoint_from_bytes(&minimal_model_bytes(), &key_map).unwrap();

        assert_eq!(checkpoint.len(), 1);
        let tensor = checkpoint.get("encoder.weight").unwrap();
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tensor.original_key, "model.encoder.weight");
    }

    #[test]
    fn test_invalid_onnx_bytes_return_invalid_model() {
        let result = OnnxModelInfo::from_bytes(b"not-an-onnx-model");
        assert!(matches!(result, Err(OnnxError::InvalidModel { .. })));
    }

    #[test]
    fn test_onnx_key_map_explicit_remap() {
        let map = OnnxKeyMap::new().with_remap("encoder.layer.0.weight", "blocks.0.weight");
        assert_eq!(
            map.map_key("encoder.layer.0.weight"),
            Some("blocks.0.weight".to_string())
        );
    }

    #[test]
    fn test_onnx_key_map_strip_prefix() {
        let map = OnnxKeyMap::new().with_strip_prefix("model.");
        assert_eq!(
            map.map_key("model.encoder.weight"),
            Some("encoder.weight".to_string())
        );
    }

    #[test]
    fn test_onnx_key_map_passthrough() {
        let map = OnnxKeyMap::new();
        assert_eq!(map.map_key("some.key"), Some("some.key".to_string()));
    }

    #[test]
    fn test_onnx_dtype_equality() {
        assert_eq!(OnnxDtype::Float32, OnnxDtype::Float32);
        assert_ne!(OnnxDtype::Float16, OnnxDtype::BFloat16);
    }

    #[test]
    fn test_onnx_tensor_info() {
        let info = OnnxTensorInfo {
            name: "input".to_string(),
            shape: vec![1, 3, 224, 224],
            dtype: OnnxDtype::Float32,
        };
        assert_eq!(info.shape.len(), 4);
        assert_eq!(info.dtype, OnnxDtype::Float32);
    }

    #[test]
    fn test_onnx_model_info() {
        let info = OnnxModelInfo {
            name: "test_model".to_string(),
            producer: "pytorch".to_string(),
            opset_version: 17,
            inputs: vec![OnnxTensorInfo {
                name: "input".to_string(),
                shape: vec![1, 3, 224, 224],
                dtype: OnnxDtype::Float32,
            }],
            outputs: vec![OnnxTensorInfo {
                name: "output".to_string(),
                shape: vec![1, 196, 768],
                dtype: OnnxDtype::Float32,
            }],
        };
        assert_eq!(info.inputs.len(), 1);
        assert_eq!(info.outputs.len(), 1);
        assert_eq!(info.opset_version, 17);
    }

    #[test]
    fn test_onnx_error_display() {
        let err = OnnxError::ShapeMismatch {
            name: "weight".to_string(),
            expected: vec![768, 768],
            actual: vec![512, 512],
        };
        let msg = format!("{err}");
        assert!(msg.contains("shape mismatch"));
        assert!(msg.contains("weight"));
    }
}
