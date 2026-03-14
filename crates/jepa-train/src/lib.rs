//! # jepa-train
//!
//! Training loop orchestration for JEPA models.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                     Training Step                        │
//! │                                                          │
//! │  1. MaskingStrategy  → context/target split              │
//! │  2. Context Encoder θ (gradients)    → s_x               │
//! │  3. Target Encoder ξ  (no gradients) → s_y               │
//! │  4. Predictor (gradients)            → ŝ_y               │
//! │  5. EnergyFn(ŝ_y, s_y)              → prediction loss    │
//! │  6. CollapseRegularizer              → regularization     │
//! │  7. EMA(θ → ξ)                       → target update      │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! The trainer computes the forward pass and returns loss terms; the
//! caller owns the optimizer (burn convention).
//!
//! **Caveat:** [`JepaComponents::forward_step`] masks tokens *after*
//! encoding. For strict pre-encoder masking, use the modality-specific
//! helpers in `jepa-vision`
//! ([`IJepa::forward_step_strict`](../jepa_vision/image/struct.IJepa.html#method.forward_step_strict),
//! [`VJepa::forward_step_strict`](../jepa_vision/video/struct.VJepa.html#method.forward_step_strict)).

pub mod checkpoint;
pub mod schedule;
pub mod step;
pub mod trainer;

pub use checkpoint::CheckpointMeta;
pub use schedule::{ConstantSchedule, LrSchedule, WarmupCosineSchedule};
pub use step::{TrainConfig, TrainMetrics, TrainStepOutput};
pub use trainer::{schedule_values, JepaComponents, JepaForwardOutput, StepReport};
