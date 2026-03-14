//! Training step types, configuration, and metrics.
//!
//! - [`TrainStepOutput`] — decomposed loss terms from a single step.
//! - [`TrainConfig`] — validated hyperparameters (LR, momentum, batch size).
//! - [`TrainMetrics`] — running averages for logging windows.

use burn::tensor::{backend::Backend, Tensor};

use jepa_core::types::Energy;

/// Output of a single training step.
///
/// Contains the decomposed loss terms for logging and monitoring.
#[derive(Debug, Clone)]
pub struct TrainStepOutput<B: Backend> {
    /// Prediction energy (main loss signal). Shape: `[1]`
    pub energy: Energy<B>,
    /// Collapse prevention regularization loss. Shape: `[1]`
    pub regularization: Tensor<B, 1>,
    /// Total loss (energy + weighted regularization). Shape: `[1]`
    pub total_loss: Tensor<B, 1>,
    /// Training step number.
    pub step: usize,
    /// Learning rate used for this step.
    pub learning_rate: f64,
    /// EMA momentum used for this step.
    pub ema_momentum: f64,
}

/// Configuration for the JEPA training loop.
///
/// # Example
///
/// ```
/// use jepa_train::TrainConfig;
///
/// let config = TrainConfig::default();
/// assert!(config.validate().is_ok());
///
/// // Custom config
/// let config = TrainConfig {
///     total_steps: 50_000,
///     warmup_steps: 5_000,
///     peak_lr: 1e-3,
///     ..TrainConfig::default()
/// };
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainConfig {
    /// Total number of training steps.
    pub total_steps: usize,
    /// Number of warmup steps for the learning rate schedule.
    pub warmup_steps: usize,
    /// Peak learning rate.
    pub peak_lr: f64,
    /// Weight for the collapse prevention loss.
    pub regularization_weight: f64,
    /// Base EMA momentum.
    pub ema_momentum: f64,
    /// Batch size.
    pub batch_size: usize,
    /// Log training metrics every N steps.
    pub log_interval: usize,
    /// Save checkpoint every N steps.
    pub checkpoint_interval: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            total_steps: 100_000,
            warmup_steps: 10_000,
            peak_lr: 1.5e-4,
            regularization_weight: 1.0,
            ema_momentum: 0.996,
            batch_size: 256,
            log_interval: 100,
            checkpoint_interval: 5_000,
        }
    }
}

impl TrainConfig {
    /// Validate the training configuration.
    pub fn validate(&self) -> Result<(), TrainConfigError> {
        if self.total_steps == 0 {
            return Err(TrainConfigError::ZeroTotalSteps);
        }
        if self.warmup_steps > self.total_steps {
            return Err(TrainConfigError::WarmupExceedsTotal {
                warmup: self.warmup_steps,
                total: self.total_steps,
            });
        }
        if self.peak_lr <= 0.0 {
            return Err(TrainConfigError::InvalidLr(self.peak_lr));
        }
        if !(0.0..=1.0).contains(&self.ema_momentum) {
            return Err(TrainConfigError::InvalidMomentum(self.ema_momentum));
        }
        if self.batch_size == 0 {
            return Err(TrainConfigError::ZeroBatchSize);
        }
        Ok(())
    }
}

/// Errors from training configuration validation.
#[derive(Debug, thiserror::Error)]
pub enum TrainConfigError {
    #[error("total_steps must be positive")]
    ZeroTotalSteps,
    #[error("warmup_steps ({warmup}) exceeds total_steps ({total})")]
    WarmupExceedsTotal { warmup: usize, total: usize },
    #[error("peak_lr must be positive, got {0}")]
    InvalidLr(f64),
    #[error("ema_momentum must be in [0.0, 1.0], got {0}")]
    InvalidMomentum(f64),
    #[error("batch_size must be positive")]
    ZeroBatchSize,
}

/// Training metrics aggregated over multiple steps.
///
/// # Example
///
/// ```
/// use jepa_train::TrainMetrics;
///
/// let mut metrics = TrainMetrics::default();
/// metrics.record(1.0, 0.5, 1.5);
/// metrics.record(2.0, 1.0, 3.0);
///
/// let (avg_energy, avg_reg, avg_total) = metrics.take_averages();
/// assert!((avg_energy - 1.5).abs() < 1e-10);
/// assert!((avg_total - 2.25).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Default)]
pub struct TrainMetrics {
    /// Running sum of energy values.
    pub energy_sum: f64,
    /// Running sum of regularization values.
    pub regularization_sum: f64,
    /// Running sum of total loss values.
    pub total_loss_sum: f64,
    /// Number of steps aggregated.
    pub count: usize,
}

impl TrainMetrics {
    /// Record metrics from a training step.
    pub fn record(&mut self, energy: f64, regularization: f64, total_loss: f64) {
        self.energy_sum += energy;
        self.regularization_sum += regularization;
        self.total_loss_sum += total_loss;
        self.count += 1;
    }

    /// Get average metrics and reset.
    pub fn take_averages(&mut self) -> (f64, f64, f64) {
        if self.count == 0 {
            return (0.0, 0.0, 0.0);
        }
        let n = self.count as f64;
        let avgs = (
            self.energy_sum / n,
            self.regularization_sum / n,
            self.total_loss_sum / n,
        );
        *self = Self::default();
        avgs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        assert!(TrainConfig::default().validate().is_ok());
    }

    #[test]
    fn test_zero_total_steps_rejected() {
        let config = TrainConfig {
            total_steps: 0,
            ..TrainConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(TrainConfigError::ZeroTotalSteps)
        ));
    }

    #[test]
    fn test_warmup_exceeds_total_rejected() {
        let config = TrainConfig {
            warmup_steps: 200,
            total_steps: 100,
            ..TrainConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(TrainConfigError::WarmupExceedsTotal { .. })
        ));
    }

    #[test]
    fn test_invalid_lr_rejected() {
        let config = TrainConfig {
            peak_lr: -1.0,
            ..TrainConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(TrainConfigError::InvalidLr(_))
        ));
    }

    #[test]
    fn test_invalid_momentum_rejected() {
        let config = TrainConfig {
            ema_momentum: 1.5,
            ..TrainConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(TrainConfigError::InvalidMomentum(_))
        ));
    }

    #[test]
    fn test_train_metrics() {
        let mut metrics = TrainMetrics::default();
        metrics.record(1.0, 0.5, 1.5);
        metrics.record(2.0, 1.0, 3.0);

        let (e, r, t) = metrics.take_averages();
        assert!((e - 1.5).abs() < 1e-10);
        assert!((r - 0.75).abs() < 1e-10);
        assert!((t - 2.25).abs() < 1e-10);

        // After take_averages, metrics should be reset
        assert_eq!(metrics.count, 0);
    }

    #[test]
    fn test_train_metrics_empty() {
        let mut metrics = TrainMetrics::default();
        let (e, r, t) = metrics.take_averages();
        assert_eq!(e, 0.0);
        assert_eq!(r, 0.0);
        assert_eq!(t, 0.0);
    }

    #[test]
    fn test_config_serialization() {
        let config = TrainConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: TrainConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.total_steps, config.total_steps);
        assert_eq!(deserialized.batch_size, config.batch_size);
    }
}
