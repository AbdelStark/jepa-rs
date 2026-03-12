//! # jepa-vision
//!
//! Vision-specific JEPA implementations.
//!
//! Provides Vision Transformer (ViT) encoder, patch embedding, and
//! rotary position encoding for images and video. Implements
//! I-JEPA and V-JEPA architectures using core traits from `jepa-core`.
//!
//! ## Modules
//! - [`patch`] — Image patchification and linear projection
//! - [`rope`] — 2D Rotary Position Embedding for spatial awareness
//! - [`vit`] — Vision Transformer encoder
//! - [`image`] — I-JEPA image pipeline and transformer predictor

pub mod image;
pub mod patch;
pub mod rope;
pub mod vit;
