//! ONNX model import for JEPA models.
//!
//! Provides types and traits for loading JEPA models from ONNX format.
//! ONNX is an open format for ML model interchange, enabling jepa-rs to
//! load models exported from PyTorch, TensorFlow, and other frameworks.
//!
//! ## Status
//!
//! This module defines the API surface for ONNX loading. The actual
//! loading logic requires an ONNX runtime dependency (e.g., `ort` crate)
//! and will be implemented when the dependency is added.
//!
//! ## Usage (future)
//!
//! ```ignore
//! let model_info = OnnxModelInfo::from_file("model.onnx")?;
//! println!("Inputs: {:?}", model_info.inputs);
//! println!("Outputs: {:?}", model_info.outputs);
//! ```

use std::collections::HashMap;
use std::path::Path;

/// Metadata about an ONNX model.
///
/// Contains information extracted from the ONNX model graph
/// without loading the full weights into memory.
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
    /// ONNX runtime is not available.
    #[error("ONNX runtime not available: add `ort` dependency to enable ONNX loading")]
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
        // Check explicit remappings first
        if let Some(mapped) = self.remappings.get(onnx_key) {
            return Some(mapped.clone());
        }

        // Strip prefix if configured
        let key = match &self.strip_prefix {
            Some(prefix) => onnx_key.strip_prefix(prefix.as_str()).unwrap_or(onnx_key),
            None => onnx_key,
        };

        Some(key.to_string())
    }
}

impl OnnxModelInfo {
    /// Load model info from an ONNX file path.
    ///
    /// This is a placeholder that returns an error until the ONNX runtime
    /// dependency is added.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, OnnxError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(OnnxError::FileNotFound {
                path: path.display().to_string(),
            });
        }

        Err(OnnxError::RuntimeNotAvailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_onnx_file_returns_file_not_found() {
        let result = OnnxModelInfo::from_file("nonexistent.onnx");
        assert!(matches!(result, Err(OnnxError::FileNotFound { .. })));
    }

    #[test]
    fn test_existing_onnx_file_returns_runtime_not_available() {
        let path = std::env::temp_dir().join(format!("jepa-rs-test-{}.onnx", std::process::id()));
        std::fs::write(&path, b"fake-onnx-model").expect("temp file should be writable");

        let result = OnnxModelInfo::from_file(&path);
        assert!(matches!(result, Err(OnnxError::RuntimeNotAvailable)));

        std::fs::remove_file(path).expect("temp file should be removable");
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
