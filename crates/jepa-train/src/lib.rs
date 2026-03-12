//! # jepa-train
//!
//! Training loop utilities for JEPA models.
//!
//! Implements RFC-008 (Training Loop) with learning rate schedules,
//! checkpoint management, and training step orchestration.
//!
//! ## Modules
//! - [`schedule`] — Learning rate schedules (warmup + cosine decay)
//! - [`checkpoint`] — Checkpoint metadata for save/resume
//! - [`step`] — Training step types, config, and metrics

pub mod checkpoint;
pub mod schedule;
pub mod step;

pub use checkpoint::CheckpointMeta;
pub use schedule::{ConstantSchedule, LrSchedule, WarmupCosineSchedule};
pub use step::{TrainConfig, TrainMetrics, TrainStepOutput};
