//! # jepa-compat
//!
//! Checkpoint compatibility and ONNX runtime for jepa-rs.
//!
//! Bridges the reference Python implementations (I-JEPA, V-JEPA, V-JEPA 2)
//! with jepa-rs models through multiple loading paths:
//!
//! - **SafeTensors** — Load weights directly from `.safetensors` checkpoints
//! - **ONNX metadata** — Inspect ONNX model structure and extract initializers
//! - **ONNX runtime** — Execute ONNX models via the tract inference engine
//! - **Model registry** — Discover pretrained Facebook Research JEPA models
//!
//! ## Quick start: ONNX inference
//!
//! ```no_run
//! use jepa_compat::runtime::OnnxSession;
//!
//! let session = OnnxSession::from_path("ijepa_encoder.onnx")?;
//! let input = vec![0.0f32; 1 * 3 * 224 * 224];
//! let output = session.run_f32(&[1, 3, 224, 224], &input)?;
//! println!("Output shape: {:?}", output.shape);
//! # Ok::<(), jepa_compat::runtime::RuntimeError>(())
//! ```
//!
//! ## Quick start: pretrained model discovery
//!
//! ```
//! use jepa_compat::registry::list_models;
//!
//! for model in list_models() {
//!     println!("{}: {} ({})", model.name, model.param_count_human(), model.architecture);
//! }
//! ```

pub mod keymap;
pub mod onnx;
pub mod registry;
pub mod runtime;
pub mod safetensors;
