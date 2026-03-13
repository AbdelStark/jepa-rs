//! # jepa-train
//!
//! Training loop orchestration for JEPA models.
//!
//! Implements RFC-008 (Training Loop). This crate ties together all the
//! pieces defined in [`jepa_core`] into a single training step:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                        Training Step                               │
//! │                                                                     │
//! │  1. MaskingStrategy  → context / target split                       │
//! │  2. Context Encoder (θ, gradients) → s_x                            │
//! │  3. Target Encoder  (ξ, no grad)   → s_y                            │
//! │  4. Predictor       (gradients)    → ŝ_y                            │
//! │  5. EnergyFn(ŝ_y, s_y)            → prediction loss                │
//! │  6. CollapseRegularizer            → regularization loss            │
//! │  7. EMA(θ → ξ)                     → target encoder update          │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! The trainer computes the forward pass and returns loss terms; the caller
//! owns the optimizer (following burn's convention of separating model
//! logic from optimization).
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`step`] | [`TrainStepOutput`], [`TrainConfig`] (with validation), [`TrainMetrics`] |
//! | [`schedule`] | [`LrSchedule`] trait, [`WarmupCosineSchedule`], [`ConstantSchedule`] |
//! | [`checkpoint`] | [`CheckpointMeta`] — save/resume training state |
//! | [`trainer`] | [`JepaComponents`] — generic forward step orchestrator |
//!
//! ## Important caveat
//!
//! [`JepaComponents::forward_step`] is a **generic** orchestration helper.
//! Because [`jepa_core::Encoder::Input`] is opaque, it cannot mask tokens
//! *before* encoder self-attention. For strict pre-encoder masking (required
//! for exact parity with the reference I-JEPA / V-JEPA implementations),
//! use the modality-specific helpers in `jepa-vision`.

pub mod checkpoint;
pub mod schedule;
pub mod step;
pub mod trainer;

pub use checkpoint::CheckpointMeta;
pub use schedule::{ConstantSchedule, LrSchedule, WarmupCosineSchedule};
pub use step::{TrainConfig, TrainMetrics, TrainStepOutput};
pub use trainer::{schedule_values, JepaComponents, JepaForwardOutput, StepReport};
