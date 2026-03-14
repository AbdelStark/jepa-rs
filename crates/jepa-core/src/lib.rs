//! # jepa-core
//!
//! Core traits and tensor abstractions for the
//! **Joint Embedding Predictive Architecture (JEPA)**.
//!
//! JEPA (LeCun, 2022) is a self-supervised learning framework that predicts in
//! *representation space* rather than pixel space. Instead of reconstructing raw
//! inputs (as in MAE or BERT), a JEPA model learns to predict the latent
//! representations of masked target regions from visible context regions. This
//! avoids wasting model capacity on pixel-level details and encourages the
//! encoder to capture high-level semantic structure.
//!
//! ```text
//!                   ┌────────────────┐
//!        x_context ─►  Context       │
//!                   │  Encoder (θ)   ├─► s_x ──┐
//!                   └────────────────┘         │
//!                                              ▼
//!                                        ┌──────────┐
//!                              z (opt.) ─►          │
//!                                        │ Predictor├─► ŝ_y ──┐
//!                     target_positions ─►│          │         │
//!                                        └──────────┘         │  ┌──────────┐
//!                                                             ├──► EnergyFn │─► loss
//!                   ┌────────────────┐                        │  └──────────┘
//!        x_target  ─►  Target        │                        │
//!                   │  Encoder (ξ)   ├─► s_y ─────────────────┘
//!                   └────────────────┘
//!                        ↑
//!                        │ EMA(θ → ξ)
//! ```
//!
//! This crate is **backend-agnostic**: all tensor-bearing APIs are generic over
//! [`burn::tensor::backend::Backend`], so they work with any burn backend
//! (NdArray, Wgpu, Tch, etc.).
//!
//! ## Crate layout
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`encoder`] | [`Encoder`] trait — maps raw inputs to [`Representation`]s |
//! | [`predictor`] | [`Predictor`] trait — predicts target representations from context |
//! | [`energy`] | [`EnergyFn`] trait and impls ([`L2Energy`], [`CosineEnergy`], [`SmoothL1Energy`]) |
//! | [`masking`] | [`MaskingStrategy`] trait and impls ([`BlockMasking`], [`SpatiotemporalMasking`], [`MultiBlockMasking`]) |
//! | [`collapse`] | [`CollapseRegularizer`] trait and impls ([`VICReg`], [`BarlowTwins`]) |
//! | [`ema`] | [`Ema`] — exponential moving average updater with optional cosine schedule |
//! | [`types`] | Semantic tensor wrappers: [`Representation`], [`Energy`], [`MaskSpec`], [`InputShape`] |
//! | [`config`] | [`JepaConfig`] with ViT presets and a validated [`JepaConfigBuilder`] |
//!
//! ## Quick start
//!
//! ```rust
//! use jepa_core::{Encoder, Predictor, EnergyFn, MaskingStrategy};
//! use jepa_core::types::{Representation, InputShape};
//! use jepa_core::energy::L2Energy;
//! use jepa_core::masking::BlockMasking;
//! use jepa_core::ema::Ema;
//! use rand::SeedableRng;
//!
//! // Configure masking: 4 target blocks covering ~15-20% of patches
//! let masking = BlockMasking {
//!     num_targets: 4,
//!     target_scale: (0.15, 0.2),
//!     target_aspect_ratio: (0.75, 1.5),
//! };
//!
//! // Generate a mask for a 14×14 patch grid (ViT-H/14 on 224×224)
//! let shape = InputShape::Image { height: 14, width: 14 };
//! let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
//! let mask = masking.generate_mask(&shape, &mut rng);
//! assert!(mask.validate().is_ok());
//!
//! // EMA with cosine momentum schedule
//! let ema = Ema::with_cosine_schedule(0.996, 100_000);
//! assert!((ema.get_momentum(0) - 0.996).abs() < 1e-6);
//! ```
//!
//! ## References
//!
//! - LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*.
//! - Assran, M. et al. (2023). *Self-Supervised Learning from Images with a
//!   Joint-Embedding Predictive Architecture*. CVPR.
//! - Bardes, A. et al. (2024). *V-JEPA: Latent Video Prediction for Visual
//!   Representation Learning*.
//! - Bardes, A. et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable
//!   Understanding, Generation, and Planning*.

pub mod collapse;
pub mod config;
pub mod ema;
pub mod encoder;
pub mod energy;
pub mod masking;
pub mod predictor;
pub mod types;

// --- Types ---
pub use types::{Energy, InputShape, MaskError, MaskSpec, Representation};

// --- Traits ---
pub use collapse::CollapseRegularizer;
pub use encoder::Encoder;
pub use energy::EnergyFn;
pub use masking::MaskingStrategy;
pub use predictor::Predictor;

// --- Configuration ---
pub use config::{ConfigError, JepaConfig, JepaConfigBuilder};

// --- Concrete implementations ---
pub use collapse::{BarlowTwins, VICReg};
pub use ema::{CosineMomentumSchedule, Ema};
pub use energy::{CosineEnergy, L2Energy, SmoothL1Energy};
pub use masking::{BlockMasking, MultiBlockMasking, SpatiotemporalMasking};
