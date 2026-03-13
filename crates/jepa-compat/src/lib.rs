//! # jepa-compat
//!
//! Checkpoint compatibility, model registry, and ONNX runtime for jepa-rs.
//!
//! This crate bridges the reference Python implementations (I-JEPA, V-JEPA,
//! V-JEPA 2) with jepa-rs by providing multiple checkpoint loading paths
//! and a pretrained model registry.
//!
//! ## Loading paths
//!
//! | Path | Format | Status |
//! |------|--------|--------|
//! | [`safetensors`] | `.safetensors` (F32 / F16 / BF16) | Functional |
//! | [`onnx`] | ONNX metadata + initializer extraction | Functional |
//! | [`runtime`] | ONNX graph execution via tract | Functional |
//! | [`keymap`] | PyTorch → burn key remapping | Functional |
//! | [`registry`] | Pretrained model discovery | Functional |
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
