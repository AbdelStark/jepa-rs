//! Checkpoint save/load types for JEPA training.
//!
//! Implements RFC-008 (Training Loop) — checkpoint component.
//!
//! Checkpointing allows resuming training from a saved state.
//! A checkpoint contains the training step, learning rate schedule
//! position, and EMA momentum state.

use serde::{Deserialize, Serialize};

/// Metadata stored alongside a checkpoint.
///
/// This captures the training state at the time the checkpoint was saved,
/// allowing training to resume from exactly where it left off.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    /// Training step at which this checkpoint was saved.
    pub step: usize,
    /// Current learning rate at checkpoint time.
    pub learning_rate: f64,
    /// Current EMA momentum at checkpoint time.
    pub ema_momentum: f64,
    /// Total training steps configured.
    pub total_steps: usize,
    /// Total loss at checkpoint time (for logging).
    pub last_loss: Option<f64>,
}

impl CheckpointMeta {
    /// Create metadata for a new training run (step 0).
    pub fn new(total_steps: usize) -> Self {
        Self {
            step: 0,
            learning_rate: 0.0,
            ema_momentum: 0.996,
            total_steps,
            last_loss: None,
        }
    }

    /// Whether training is complete.
    pub fn is_complete(&self) -> bool {
        self.step >= self.total_steps
    }

    /// Training progress as a fraction in `[0.0, 1.0]`.
    pub fn progress(&self) -> f64 {
        if self.total_steps == 0 {
            1.0
        } else {
            self.step as f64 / self.total_steps as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_meta_new() {
        let meta = CheckpointMeta::new(10000);
        assert_eq!(meta.step, 0);
        assert_eq!(meta.total_steps, 10000);
        assert!(!meta.is_complete());
    }

    #[test]
    fn test_checkpoint_meta_progress() {
        let mut meta = CheckpointMeta::new(100);
        assert!((meta.progress() - 0.0).abs() < 1e-10);

        meta.step = 50;
        assert!((meta.progress() - 0.5).abs() < 1e-10);

        meta.step = 100;
        assert!((meta.progress() - 1.0).abs() < 1e-10);
        assert!(meta.is_complete());
    }

    #[test]
    fn test_checkpoint_meta_serialization() {
        let meta = CheckpointMeta {
            step: 5000,
            learning_rate: 1e-4,
            ema_momentum: 0.998,
            total_steps: 10000,
            last_loss: Some(0.42),
        };

        let json = serde_json::to_string(&meta).unwrap();
        let deserialized: CheckpointMeta = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.step, 5000);
        assert!((deserialized.learning_rate - 1e-4).abs() < 1e-10);
        assert_eq!(deserialized.last_loss, Some(0.42));
    }
}
