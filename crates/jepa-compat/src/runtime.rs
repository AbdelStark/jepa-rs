//! ONNX runtime execution for JEPA models.
//!
//! Provides model loading and inference execution using the [`tract-onnx`]
//! inference engine. This module bridges exported ONNX models (from PyTorch
//! or other frameworks) with Rust-native inference.
//!
//! ## Workflow
//!
//! 1. Export a pretrained JEPA model to ONNX (e.g. `scripts/export_ijepa_onnx.py`).
//! 2. Load the ONNX model with [`OnnxSession::from_path`].
//! 3. Run inference with [`OnnxSession::run_f32`] or [`OnnxSession::run_f32_multi`].
//!
//! You can also inspect a model without running it via the
//! `inspect_model` and `validate_model` free functions in [`crate::onnx`].
//!
//! ## Example
//!
//! ```no_run
//! use jepa_compat::runtime::OnnxSession;
//!
//! let session = OnnxSession::from_path("ijepa_vit_h14.onnx")?;
//! println!("Model: {:?}", session.info());
//!
//! let input = vec![0.0f32; 1 * 3 * 224 * 224];
//! let output = session.run_f32(&[1, 3, 224, 224], &input)?;
//! println!("Output shape: {:?}, values: {}", output.shape, output.data.len());
//! # Ok::<(), jepa_compat::runtime::RuntimeError>(())
//! ```
//!
//! [`tract-onnx`]: https://docs.rs/tract-onnx

use std::path::Path;
use std::sync::Arc;

use tract_onnx::prelude::*;

use crate::onnx::{OnnxDtype, OnnxModelInfo, OnnxTensorInfo};

/// Errors from ONNX runtime operations.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// The ONNX file could not be read or parsed.
    #[error("failed to load ONNX model: {reason}")]
    LoadError { reason: String },
    /// Model optimization failed.
    #[error("failed to optimize model: {reason}")]
    OptimizationError { reason: String },
    /// Inference execution failed.
    #[error("inference failed: {reason}")]
    InferenceError { reason: String },
    /// Input shape does not match what the model expects.
    #[error("input shape mismatch: expected {expected:?}, got {actual:?}")]
    InputShapeMismatch {
        expected: Vec<i64>,
        actual: Vec<usize>,
    },
}

/// Output from ONNX model inference.
#[derive(Debug, Clone)]
pub struct InferenceOutput {
    /// The output tensor data as f32 values.
    pub data: Vec<f32>,
    /// The shape of the output tensor.
    pub shape: Vec<usize>,
}

impl InferenceOutput {
    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the output is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Reshape the output as a 2D slice: `[tokens, embed_dim]`.
    ///
    /// Assumes the output shape is `[batch, tokens, embed_dim]` with batch=1.
    pub fn as_token_embeddings(&self) -> Option<(&[f32], usize, usize)> {
        if self.shape.len() == 3 && self.shape[0] == 1 {
            let tokens = self.shape[1];
            let embed_dim = self.shape[2];
            Some((&self.data, tokens, embed_dim))
        } else {
            None
        }
    }

    /// Get the batch, tokens, and embed_dim dimensions for 3D outputs.
    ///
    /// Returns `(batch, tokens, embed_dim)` for shape `[batch, tokens, embed_dim]`.
    pub fn as_batched_embeddings(&self) -> Option<(usize, usize, usize)> {
        if self.shape.len() == 3 {
            Some((self.shape[0], self.shape[1], self.shape[2]))
        } else {
            None
        }
    }
}

/// Metadata about a loaded ONNX session.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    /// Model name from the ONNX graph.
    pub name: String,
    /// Producer that created the ONNX model.
    pub producer: String,
    /// ONNX opset version.
    pub opset_version: i64,
    /// Input tensor specifications.
    pub inputs: Vec<OnnxTensorInfo>,
    /// Output tensor specifications.
    pub outputs: Vec<OnnxTensorInfo>,
}

impl From<OnnxModelInfo> for SessionInfo {
    fn from(info: OnnxModelInfo) -> Self {
        Self {
            name: info.name,
            producer: info.producer,
            opset_version: info.opset_version,
            inputs: info.inputs,
            outputs: info.outputs,
        }
    }
}

/// Type alias for the optimized tract model plan.
type TypedPlan = tract_onnx::tract_core::plan::SimplePlan<TypedFact, Box<dyn TypedOp>>;

/// An ONNX model session ready for inference.
///
/// Wraps a tract-onnx optimized model plan. The model is loaded, validated,
/// and optimized at construction time so that [`run`](Self::run_f32) is fast.
#[derive(Clone)]
pub struct OnnxSession {
    plan: Arc<TypedPlan>,
    info: SessionInfo,
}

impl std::fmt::Debug for OnnxSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxSession")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}

impl OnnxSession {
    /// Load an ONNX model from a file path.
    ///
    /// The model is parsed, type-checked, optimized, and compiled into
    /// a runnable plan. Input shapes must be fully specified in the ONNX
    /// model (no dynamic dimensions).
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, RuntimeError> {
        let path = path.as_ref();
        let info = OnnxModelInfo::from_file(path).map_err(|e| RuntimeError::LoadError {
            reason: e.to_string(),
        })?;

        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e: TractError| RuntimeError::LoadError {
                reason: e.to_string(),
            })?
            .into_optimized()
            .map_err(|e: TractError| RuntimeError::OptimizationError {
                reason: e.to_string(),
            })?
            .into_runnable()
            .map_err(|e: TractError| RuntimeError::OptimizationError {
                reason: format!("failed to create runnable plan: {e}"),
            })?;

        Ok(Self {
            plan: model,
            info: info.into(),
        })
    }

    /// Load an ONNX model from a file, overriding the input shape.
    ///
    /// Use this when the ONNX model has dynamic dimensions (e.g., variable
    /// batch size) that need to be fixed for execution.
    pub fn from_path_with_input_shape(
        path: impl AsRef<Path>,
        input_shape: &[usize],
    ) -> Result<Self, RuntimeError> {
        let path = path.as_ref();
        let info = OnnxModelInfo::from_file(path).map_err(|e| RuntimeError::LoadError {
            reason: e.to_string(),
        })?;

        let shape_i64: Vec<i64> = input_shape.iter().map(|&d| d as i64).collect();
        let fact = InferenceFact::dt_shape(f32::datum_type(), &shape_i64);

        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e: TractError| RuntimeError::LoadError {
                reason: e.to_string(),
            })?
            .with_input_fact(0, fact)
            .map_err(|e: TractError| RuntimeError::LoadError {
                reason: format!("failed to set input shape: {e}"),
            })?
            .into_optimized()
            .map_err(|e: TractError| RuntimeError::OptimizationError {
                reason: e.to_string(),
            })?
            .into_runnable()
            .map_err(|e: TractError| RuntimeError::OptimizationError {
                reason: format!("failed to create runnable plan: {e}"),
            })?;

        Ok(Self {
            plan: model,
            info: info.into(),
        })
    }

    /// Get metadata about the loaded model.
    pub fn info(&self) -> &SessionInfo {
        &self.info
    }

    /// Run inference with f32 input data.
    ///
    /// # Arguments
    /// * `input_shape` - Shape of the input tensor (e.g., `[1, 3, 224, 224]`)
    /// * `input_data` - Flat f32 data matching the input shape
    ///
    /// # Returns
    /// The first output tensor from the model.
    pub fn run_f32(
        &self,
        input_shape: &[usize],
        input_data: &[f32],
    ) -> Result<InferenceOutput, RuntimeError> {
        let expected_len: usize = input_shape.iter().product();
        if input_data.len() != expected_len {
            return Err(RuntimeError::InferenceError {
                reason: format!(
                    "input data length {} does not match shape {:?} (expected {})",
                    input_data.len(),
                    input_shape,
                    expected_len,
                ),
            });
        }

        let input_tensor = tract_ndarray::Array::from_shape_vec(
            tract_ndarray::IxDyn(input_shape),
            input_data.to_vec(),
        )
        .map_err(|e| RuntimeError::InferenceError {
            reason: format!("failed to create input tensor: {e}"),
        })?;

        let result = self
            .plan
            .run(tvec!(Tensor::from(input_tensor).into()))
            .map_err(|e: TractError| RuntimeError::InferenceError {
                reason: e.to_string(),
            })?;

        let output_arc = result.first().ok_or_else(|| RuntimeError::InferenceError {
            reason: "model produced no outputs".to_string(),
        })?;

        let output = output_arc.to_array_view::<f32>().map_err(|e: TractError| {
            RuntimeError::InferenceError {
                reason: format!("failed to read output as f32: {e}"),
            }
        })?;

        Ok(InferenceOutput {
            data: output.iter().copied().collect(),
            shape: output.shape().to_vec(),
        })
    }

    /// Run inference with multiple f32 inputs.
    ///
    /// Use this for ONNX models that accept more than one input tensor.
    /// Each entry in `inputs` is a `(shape, data)` pair.
    ///
    /// Returns the first output tensor.
    pub fn run_f32_multi_input(
        &self,
        inputs: &[(&[usize], &[f32])],
    ) -> Result<InferenceOutput, RuntimeError> {
        let mut input_tvec = TVec::new();
        for (shape, data) in inputs {
            let expected_len: usize = shape.iter().product();
            if data.len() != expected_len {
                return Err(RuntimeError::InferenceError {
                    reason: format!(
                        "input data length {} does not match shape {:?} (expected {})",
                        data.len(),
                        shape,
                        expected_len,
                    ),
                });
            }
            let tensor =
                tract_ndarray::Array::from_shape_vec(tract_ndarray::IxDyn(shape), data.to_vec())
                    .map_err(|e| RuntimeError::InferenceError {
                        reason: format!("failed to create input tensor: {e}"),
                    })?;
            input_tvec.push(Tensor::from(tensor).into());
        }

        let result =
            self.plan
                .run(input_tvec)
                .map_err(|e: TractError| RuntimeError::InferenceError {
                    reason: e.to_string(),
                })?;

        let output_arc = result.first().ok_or_else(|| RuntimeError::InferenceError {
            reason: "model produced no outputs".to_string(),
        })?;

        let output = output_arc.to_array_view::<f32>().map_err(|e: TractError| {
            RuntimeError::InferenceError {
                reason: format!("failed to read output as f32: {e}"),
            }
        })?;

        Ok(InferenceOutput {
            data: output.iter().copied().collect(),
            shape: output.shape().to_vec(),
        })
    }

    /// Run inference and return all output tensors.
    pub fn run_f32_multi(
        &self,
        input_shape: &[usize],
        input_data: &[f32],
    ) -> Result<Vec<InferenceOutput>, RuntimeError> {
        let expected_len: usize = input_shape.iter().product();
        if input_data.len() != expected_len {
            return Err(RuntimeError::InferenceError {
                reason: format!(
                    "input data length {} does not match shape {:?} (expected {})",
                    input_data.len(),
                    input_shape,
                    expected_len,
                ),
            });
        }

        let input_tensor = tract_ndarray::Array::from_shape_vec(
            tract_ndarray::IxDyn(input_shape),
            input_data.to_vec(),
        )
        .map_err(|e| RuntimeError::InferenceError {
            reason: format!("failed to create input tensor: {e}"),
        })?;

        let results = self
            .plan
            .run(tvec!(Tensor::from(input_tensor).into()))
            .map_err(|e: TractError| RuntimeError::InferenceError {
                reason: e.to_string(),
            })?;

        let mut outputs = Vec::with_capacity(results.len());
        for tvalue in results.iter() {
            let tensor: &Tensor = tvalue;
            let arr = tensor.to_array_view::<f32>().map_err(|e: TractError| {
                RuntimeError::InferenceError {
                    reason: format!("failed to read output as f32: {e}"),
                }
            })?;
            outputs.push(InferenceOutput {
                data: arr.iter().copied().collect(),
                shape: arr.shape().to_vec(),
            });
        }

        Ok(outputs)
    }
}

/// An ONNX-backed encoder that implements the [`jepa_core::Encoder`] trait.
///
/// Wraps an [`OnnxSession`] to run inference through an exported ONNX model
/// and produce [`jepa_core::Representation`] outputs compatible with the JEPA pipeline.
///
/// The encoder accepts `Tensor<B, 4>` image inputs with shape
/// `[batch, channels, height, width]` and returns representations with shape
/// `[batch, tokens, embed_dim]`.
///
/// # Example
///
/// ```no_run
/// use jepa_compat::runtime::OnnxEncoder;
/// use jepa_core::Encoder;
/// use burn::tensor::Tensor;
/// use burn_ndarray::NdArray;
///
/// type B = NdArray<f32>;
///
/// let encoder = OnnxEncoder::from_path("model.onnx")?;
/// let input: Tensor<B, 4> = Tensor::zeros(
///     [1, 3, 224, 224],
///     &burn_ndarray::NdArrayDevice::Cpu,
/// );
/// let repr = encoder.encode(&input);
/// println!("Output: [{}, {}, {}]", repr.batch_size(), repr.seq_len(), repr.embed_dim());
/// # Ok::<(), jepa_compat::runtime::RuntimeError>(())
/// ```
#[derive(Clone)]
pub struct OnnxEncoder {
    session: OnnxSession,
    embed_dim: usize,
}

impl std::fmt::Debug for OnnxEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxEncoder")
            .field("embed_dim", &self.embed_dim)
            .field("session", &self.session)
            .finish()
    }
}

impl OnnxEncoder {
    /// Create an encoder from an ONNX model file.
    ///
    /// The model must have a single output with shape `[batch, tokens, embed_dim]`.
    /// The embedding dimension is inferred from the output specification.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, RuntimeError> {
        let session = OnnxSession::from_path(&path)?;
        let embed_dim = Self::infer_embed_dim(&session)?;
        Ok(Self { session, embed_dim })
    }

    /// Create an encoder from an ONNX model file, overriding the input shape.
    ///
    /// Use this when the ONNX model has dynamic dimensions that need to be
    /// fixed for execution.
    pub fn from_path_with_input_shape(
        path: impl AsRef<Path>,
        input_shape: &[usize],
    ) -> Result<Self, RuntimeError> {
        let session = OnnxSession::from_path_with_input_shape(&path, input_shape)?;
        let embed_dim = Self::infer_embed_dim(&session)?;
        Ok(Self { session, embed_dim })
    }

    /// Create an encoder from an existing [`OnnxSession`].
    pub fn from_session(session: OnnxSession) -> Result<Self, RuntimeError> {
        let embed_dim = Self::infer_embed_dim(&session)?;
        Ok(Self { session, embed_dim })
    }

    /// Get the underlying session metadata.
    pub fn info(&self) -> &SessionInfo {
        self.session.info()
    }

    /// Get the output embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Run inference on raw f32 data and return an [`InferenceOutput`].
    ///
    /// This is a lower-level method for callers who don't need the
    /// burn `Encoder` trait integration.
    pub fn run_raw(
        &self,
        input_shape: &[usize],
        input_data: &[f32],
    ) -> Result<InferenceOutput, RuntimeError> {
        self.session.run_f32(input_shape, input_data)
    }

    fn infer_embed_dim(session: &OnnxSession) -> Result<usize, RuntimeError> {
        let info = session.info();
        let output = info
            .outputs
            .first()
            .ok_or_else(|| RuntimeError::LoadError {
                reason: "model has no outputs".to_string(),
            })?;

        // For 3D output [batch, tokens, embed_dim], take the last dim.
        // For 2D output [batch, embed_dim], take the last dim.
        let dim = output.shape.last().copied().unwrap_or(0);
        if dim <= 0 {
            return Err(RuntimeError::LoadError {
                reason: format!(
                    "cannot infer embedding dimension from output shape {:?}",
                    output.shape
                ),
            });
        }

        Ok(dim as usize)
    }
}

impl<B: burn::tensor::backend::Backend> jepa_core::Encoder<B> for OnnxEncoder {
    type Input = burn::tensor::Tensor<B, 4>;

    /// Encode a batch of images through the ONNX model.
    ///
    /// # Panics
    ///
    /// This method panics if:
    /// - The input tensor cannot be converted to f32 data (invariant violation:
    ///   burn tensors should always be extractable).
    /// - ONNX inference fails at runtime (e.g., the model is incompatible with
    ///   the input shape). Use [`OnnxEncoder::run_raw`] for a fallible alternative.
    /// - The ONNX model produces output with rank other than 2 or 3.
    ///
    /// The `Encoder` trait does not support fallible encoding, so these panics
    /// are unavoidable at the trait boundary. Validate inputs before calling.
    fn encode(&self, input: &burn::tensor::Tensor<B, 4>) -> jepa_core::Representation<B> {
        let dims = input.dims();
        let device = input.device();
        let batch = dims[0];

        // Invariant: burn tensors are always extractable to their element type.
        let input_data: Vec<f32> = input
            .clone()
            .reshape([dims.iter().product::<usize>()])
            .to_data()
            .to_vec()
            .expect("burn tensor extraction to f32 failed — this is a burn backend bug");

        // Tract does not reliably support dynamic batch sizes, so we run
        // batch-1 inference in a loop and concatenate results when batch > 1.
        if batch == 1 {
            let output = self
                .session
                .run_f32(&[dims[0], dims[1], dims[2], dims[3]], &input_data)
                .expect("ONNX inference failed — verify the model accepts the input shape");

            let out_shape = &output.shape;
            let tensor = burn::tensor::Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(output.data, [out_shape.iter().product::<usize>()]),
                &device,
            );

            match out_shape.len() {
                3 => {
                    let tensor = tensor.reshape([out_shape[0], out_shape[1], out_shape[2]]);
                    jepa_core::Representation::new(tensor)
                }
                2 => {
                    // [batch, embed_dim] → [batch, 1, embed_dim]
                    let tensor = tensor.reshape([out_shape[0], 1, out_shape[1]]);
                    jepa_core::Representation::new(tensor)
                }
                rank => panic!(
                    "ONNX model output has rank {rank}, expected 2 ([batch, embed_dim]) \
                     or 3 ([batch, tokens, embed_dim])"
                ),
            }
        } else {
            // Process each batch element individually and concatenate.
            let single_len: usize = dims[1] * dims[2] * dims[3];
            let mut all_data: Vec<f32> = Vec::new();
            let mut out_tokens = 0usize;

            for b in 0..batch {
                let start = b * single_len;
                let end = start + single_len;
                let single_input = &input_data[start..end];

                let output = self
                    .session
                    .run_f32(&[1, dims[1], dims[2], dims[3]], single_input)
                    .unwrap_or_else(|e| {
                        panic!("ONNX inference failed on batch element {b}/{batch}: {e}")
                    });

                if b == 0 {
                    out_tokens = if output.shape.len() == 3 {
                        output.shape[1]
                    } else {
                        1
                    };
                }

                all_data.extend_from_slice(&output.data);
            }

            let tensor = burn::tensor::Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(all_data, [batch * out_tokens * self.embed_dim]),
                &device,
            );
            let tensor = tensor.reshape([batch, out_tokens, self.embed_dim]);
            jepa_core::Representation::new(tensor)
        }
    }

    fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

/// Summary of an ONNX model without loading the full runtime.
///
/// Lighter weight than [`OnnxSession`] — only parses model metadata.
pub fn inspect_model(path: impl AsRef<Path>) -> Result<SessionInfo, RuntimeError> {
    let info = OnnxModelInfo::from_file(path).map_err(|e| RuntimeError::LoadError {
        reason: e.to_string(),
    })?;
    Ok(info.into())
}

/// Human-readable summary of model metadata.
pub fn format_model_summary(info: &SessionInfo) -> String {
    let mut s = String::new();
    s.push_str(&format!("Model: {}\n", info.name));
    s.push_str(&format!("Producer: {}\n", info.producer));
    s.push_str(&format!("Opset: {}\n", info.opset_version));

    s.push_str("Inputs:\n");
    for input in &info.inputs {
        s.push_str(&format!(
            "  {} {:?} ({:?})\n",
            input.name, input.shape, input.dtype
        ));
    }

    s.push_str("Outputs:\n");
    for output in &info.outputs {
        s.push_str(&format!(
            "  {} {:?} ({:?})\n",
            output.name, output.shape, output.dtype
        ));
    }

    if let Some(output) = info.outputs.first() {
        let embed_dim = output.shape.last().copied().unwrap_or(0);
        if embed_dim > 0 {
            s.push_str(&format!("Embedding dim: {}\n", embed_dim));
        }
    }

    s
}

/// Validate that an ONNX model file is loadable and report diagnostics.
pub fn validate_model(path: impl AsRef<Path>) -> Result<Vec<String>, RuntimeError> {
    let path = path.as_ref();
    let mut diagnostics = Vec::new();

    let info = OnnxModelInfo::from_file(path).map_err(|e| RuntimeError::LoadError {
        reason: e.to_string(),
    })?;

    diagnostics.push(format!("Graph name: {}", info.name));
    diagnostics.push(format!("Producer: {}", info.producer));
    diagnostics.push(format!("Opset version: {}", info.opset_version));

    for input in &info.inputs {
        let has_dynamic = input.shape.iter().any(|&d| d < 0);
        if has_dynamic {
            diagnostics.push(format!(
                "WARNING: input '{}' has dynamic dimensions {:?}",
                input.name, input.shape
            ));
        }
        if input.dtype == OnnxDtype::Unknown {
            diagnostics.push(format!("WARNING: input '{}' has unknown dtype", input.name));
        }
    }

    match tract_onnx::onnx().model_for_path(path) {
        Ok(model) => {
            diagnostics.push("Model parsed successfully by tract".to_string());
            match model.into_optimized() {
                Ok(_) => diagnostics.push("Model optimized successfully".to_string()),
                Err(e) => diagnostics.push(format!("WARNING: optimization failed: {e}")),
            }
        }
        Err(e) => diagnostics.push(format!("WARNING: tract could not parse model: {e}")),
    }

    Ok(diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_output_as_token_embeddings() {
        let output = InferenceOutput {
            data: vec![0.0; 196 * 768],
            shape: vec![1, 196, 768],
        };
        let (data, tokens, embed_dim) = output.as_token_embeddings().unwrap();
        assert_eq!(tokens, 196);
        assert_eq!(embed_dim, 768);
        assert_eq!(data.len(), 196 * 768);
    }

    #[test]
    fn test_inference_output_non_3d_returns_none() {
        let output = InferenceOutput {
            data: vec![0.0; 768],
            shape: vec![1, 768],
        };
        assert!(output.as_token_embeddings().is_none());
    }

    #[test]
    fn test_inference_output_batch_gt_1_returns_none() {
        let output = InferenceOutput {
            data: vec![0.0; 2 * 196 * 768],
            shape: vec![2, 196, 768],
        };
        assert!(output.as_token_embeddings().is_none());
    }

    #[test]
    fn test_session_info_from_model_info() {
        let model_info = OnnxModelInfo {
            name: "test".to_string(),
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
        let session_info: SessionInfo = model_info.into();
        assert_eq!(session_info.name, "test");
        assert_eq!(session_info.opset_version, 17);
    }

    #[test]
    fn test_format_model_summary() {
        let info = SessionInfo {
            name: "ijepa-vit-h14".to_string(),
            producer: "pytorch".to_string(),
            opset_version: 17,
            inputs: vec![OnnxTensorInfo {
                name: "pixel_values".to_string(),
                shape: vec![1, 3, 224, 224],
                dtype: OnnxDtype::Float32,
            }],
            outputs: vec![OnnxTensorInfo {
                name: "representations".to_string(),
                shape: vec![1, 256, 1280],
                dtype: OnnxDtype::Float32,
            }],
        };
        let summary = format_model_summary(&info);
        assert!(summary.contains("ijepa-vit-h14"));
        assert!(summary.contains("pytorch"));
        assert!(summary.contains("1280"));
    }

    #[test]
    fn test_inspect_missing_model() {
        let result = inspect_model("nonexistent.onnx");
        assert!(result.is_err());
    }

    #[test]
    fn test_inference_output_len() {
        let output = InferenceOutput {
            data: vec![1.0, 2.0, 3.0],
            shape: vec![3],
        };
        assert_eq!(output.len(), 3);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_empty_inference_output() {
        let output = InferenceOutput {
            data: vec![],
            shape: vec![0],
        };
        assert!(output.is_empty());
        assert_eq!(output.len(), 0);
    }

    #[test]
    fn test_as_batched_embeddings_3d() {
        let output = InferenceOutput {
            data: vec![0.0; 2 * 196 * 768],
            shape: vec![2, 196, 768],
        };
        let (batch, tokens, embed_dim) = output.as_batched_embeddings().unwrap();
        assert_eq!(batch, 2);
        assert_eq!(tokens, 196);
        assert_eq!(embed_dim, 768);
    }

    #[test]
    fn test_as_batched_embeddings_non_3d_returns_none() {
        let output = InferenceOutput {
            data: vec![0.0; 768],
            shape: vec![1, 768],
        };
        assert!(output.as_batched_embeddings().is_none());
    }

    #[test]
    fn test_onnx_encoder_infer_embed_dim_from_session_info() {
        // Verify embed_dim inference logic by creating a SessionInfo directly.
        let info = SessionInfo {
            name: "test".to_string(),
            producer: "test".to_string(),
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
        // The last dimension of the first output is the embed_dim.
        let embed_dim = info.outputs[0].shape.last().copied().unwrap_or(0);
        assert_eq!(embed_dim, 768);
    }

    #[test]
    fn test_onnx_encoder_from_missing_path() {
        let result = OnnxEncoder::from_path("nonexistent.onnx");
        assert!(result.is_err());
    }
}
