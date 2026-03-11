//! Exponential Moving Average (EMA) for target encoder updates.
//!
//! Implements RFC-007 (EMA Target Encoder).
//!
//! In JEPA, the target encoder is not trained via gradient descent.
//! Instead, its weights are an EMA of the context encoder's weights.
//! This asymmetry is critical for preventing collapse.

/// Exponential Moving Average weight updater.
///
/// Updates target parameters toward online (context) parameters:
/// ```text
/// target = momentum * target + (1 - momentum) * online
/// ```
///
/// Higher momentum means the target changes more slowly,
/// providing a more stable prediction target.
pub struct Ema {
    /// Base momentum parameter. Typical values: 0.996 to 0.9999.
    pub momentum: f64,
    /// Optional momentum schedule that increases momentum during training.
    pub schedule: Option<MomentumSchedule>,
}

impl Ema {
    /// Create a new EMA with constant momentum and no schedule.
    pub fn new(momentum: f64) -> Self {
        Self {
            momentum,
            schedule: None,
        }
    }

    /// Create a new EMA with a cosine momentum schedule.
    pub fn with_cosine_schedule(base_momentum: f64, total_steps: usize) -> Self {
        Self {
            momentum: base_momentum,
            schedule: Some(MomentumSchedule::Cosine(CosineMomentumSchedule {
                base_momentum,
                final_momentum: 1.0,
                total_steps,
            })),
        }
    }

    /// Get the effective momentum at a given training step.
    pub fn get_momentum(&self, step: usize) -> f64 {
        match &self.schedule {
            Some(schedule) => schedule.get_momentum(step),
            None => self.momentum,
        }
    }

    /// Perform a single EMA step on scalar values (for testing/simple use).
    ///
    /// Returns `momentum * target + (1 - momentum) * online`.
    pub fn step(&self, target: f64, online: f64, step: usize) -> f64 {
        let m = self.get_momentum(step);
        m * target + (1.0 - m) * online
    }
}

/// Momentum schedule variants.
#[derive(Debug, Clone)]
pub enum MomentumSchedule {
    /// Cosine schedule that increases momentum from base to final over training.
    Cosine(CosineMomentumSchedule),
}

impl MomentumSchedule {
    /// Get the momentum value at a given training step.
    pub fn get_momentum(&self, step: usize) -> f64 {
        match self {
            MomentumSchedule::Cosine(s) => s.get_momentum(step),
        }
    }
}

/// Cosine momentum schedule (V-JEPA 2 style).
///
/// Momentum increases from `base_momentum` to `final_momentum` following
/// a cosine curve over `total_steps`. This provides a slow start and
/// smooth transition.
///
/// Formula:
/// ```text
/// m(t) = final - (final - base) * (1 + cos(π * t / T)) / 2
/// ```
#[derive(Debug, Clone)]
pub struct CosineMomentumSchedule {
    /// Starting momentum (e.g., 0.996).
    pub base_momentum: f64,
    /// Final momentum (typically 1.0).
    pub final_momentum: f64,
    /// Total number of training steps.
    pub total_steps: usize,
}

impl CosineMomentumSchedule {
    /// Get the momentum value at step `t`.
    pub fn get_momentum(&self, step: usize) -> f64 {
        if self.total_steps == 0 {
            return self.final_momentum;
        }
        let t = step.min(self.total_steps - 1) as f64;
        let total = self.total_steps as f64;
        let progress = t / total;
        // Cosine annealing from base to final
        self.final_momentum
            - (self.final_momentum - self.base_momentum)
                * (1.0 + (progress * std::f64::consts::PI).cos())
                / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_momentum_1_keeps_target_unchanged() {
        let ema = Ema::new(1.0);
        let result = ema.step(5.0, 10.0, 0);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_momentum_0_copies_online() {
        let ema = Ema::new(0.0);
        let result = ema.step(5.0, 10.0, 0);
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_typical_momentum() {
        let ema = Ema::new(0.996);
        let result = ema.step(0.0, 1.0, 0);
        // 0.996 * 0.0 + 0.004 * 1.0 = 0.004
        assert!((result - 0.004).abs() < 1e-10);
    }

    #[test]
    fn test_ema_converges_to_online() {
        let ema = Ema::new(0.99);
        let online = 1.0;
        let mut target = 0.0;
        for step in 0..1000 {
            target = ema.step(target, online, step);
        }
        assert!(
            (target - 1.0).abs() < 0.01,
            "expected convergence to 1.0, got {target}"
        );
    }

    #[test]
    fn test_cosine_schedule_at_start() {
        let schedule = CosineMomentumSchedule {
            base_momentum: 0.996,
            final_momentum: 1.0,
            total_steps: 10000,
        };
        let m = schedule.get_momentum(0);
        assert!(
            (m - 0.996).abs() < 1e-6,
            "expected 0.996 at step 0, got {m}"
        );
    }

    #[test]
    fn test_cosine_schedule_at_end() {
        let schedule = CosineMomentumSchedule {
            base_momentum: 0.996,
            final_momentum: 1.0,
            total_steps: 10000,
        };
        let m = schedule.get_momentum(9999);
        assert!(
            (m - 1.0).abs() < 1e-3,
            "expected ~1.0 at final step, got {m}"
        );
    }

    #[test]
    fn test_cosine_schedule_midpoint() {
        let schedule = CosineMomentumSchedule {
            base_momentum: 0.996,
            final_momentum: 1.0,
            total_steps: 10000,
        };
        let m = schedule.get_momentum(5000);
        // At midpoint, cosine schedule should be at the average
        assert!(
            m > 0.997 && m < 0.999,
            "expected ~0.998 at midpoint, got {m}"
        );
    }

    #[test]
    fn test_cosine_schedule_is_monotonically_increasing() {
        let schedule = CosineMomentumSchedule {
            base_momentum: 0.996,
            final_momentum: 1.0,
            total_steps: 1000,
        };
        let mut prev = schedule.get_momentum(0);
        for step in 1..1000 {
            let curr = schedule.get_momentum(step);
            assert!(
                curr >= prev - 1e-10,
                "schedule not monotonic at step {step}: {prev} -> {curr}"
            );
            prev = curr;
        }
    }

    #[test]
    fn test_ema_with_schedule() {
        let ema = Ema::with_cosine_schedule(0.996, 10000);
        let m0 = ema.get_momentum(0);
        let m_end = ema.get_momentum(9999);
        assert!((m0 - 0.996).abs() < 1e-6);
        assert!((m_end - 1.0).abs() < 1e-3);
    }
}
