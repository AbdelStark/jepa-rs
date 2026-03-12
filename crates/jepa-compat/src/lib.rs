//! # jepa-compat
//!
//! PyTorch checkpoint compatibility for jepa-rs.
//!
//! Provides safetensors weight loading to bridge the reference
//! Python implementations (I-JEPA, V-JEPA, V-JEPA 2) with jepa-rs models.
//!
//! ## Modules
//! - [`safetensors`] — Load weights from safetensors files
//! - [`keymap`] — Map PyTorch state_dict keys to burn module paths

pub mod keymap;
pub mod safetensors;
