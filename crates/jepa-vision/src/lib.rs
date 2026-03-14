//! # jepa-vision
//!
//! Vision Transformer (ViT) encoders and predictors for image and video JEPA.
//!
//! This crate provides the concrete vision modules that implement the
//! abstract traits defined in [`jepa_core`]:
//!
//! ```text
//!  Image / Video
//!       │
//!       ▼
//! ┌────────────┐   ┌──────────────────┐
//! │ Patch /     │──►│  ViT Encoder     │──► Representation
//! │ Tubelet     │   │  (+ 2D/3D RoPE)  │    [B, S, D]
//! │ Embedding   │   └──────────────────┘
//! └────────────┘
//! ```
//!
//! ## Modules
//!
//! | Module | Contents | Reference |
//! |--------|----------|-----------|
//! | [`patch`] | [`PatchEmbedding`](patch::PatchEmbedding) — 2D image patchification + linear projection | ViT (Dosovitskiy 2021) |
//! | [`rope`] | [`RotaryPositionEncoding2D`](rope::RotaryPositionEncoding2D) — 2D rotary position encoding | RoFormer (Su 2021) |
//! | [`vit`] | [`VitEncoder`](vit::VitEncoder) — image ViT with configurable presets (Tiny → giant) | |
//! | [`image`] | [`TransformerPredictor`](image::TransformerPredictor), [`IJepa`](image::IJepa) — I-JEPA pipeline with `forward_step_strict` | Assran et al. (2023) |
//! | [`video`] | [`VitVideoEncoder`](video::VitVideoEncoder), [`VJepa`](video::VJepa) — V-JEPA with 3D tubelets + 3D RoPE | Bardes et al. (2024) |
//! | [`slot_attention`] | [`SlotAttention`](slot_attention::SlotAttention), [`SlotEncoder`](slot_attention::SlotEncoder) — object-centric encoding for C-JEPA | Locatello (2020), Nam (2025) |
//!
//! ## Quick start
//!
//! ```rust
//! use jepa_vision::vit::VitConfig;
//! use jepa_core::Encoder;
//! use burn_ndarray::NdArray;
//!
//! type B = NdArray<f32>;
//! let device = burn_ndarray::NdArrayDevice::Cpu;
//!
//! // Tiny ViT for tests; use VitConfig::vit_base_patch16() for real workloads
//! let encoder = VitConfig::tiny_test().init::<B>(&device);
//! assert_eq!(encoder.embed_dim(), 32);
//! ```

pub mod image;
pub mod patch;
pub mod rope;
pub mod slot_attention;
pub(crate) mod token_ops;
pub mod video;
pub mod vit;
