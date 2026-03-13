//! Learning rate schedules for JEPA training.
//!
//! Implements RFC-008 (Training Loop) — schedule component.
//!
//! JEPA training typically uses a **warmup + cosine decay** schedule:
//! the LR linearly ramps from near-zero to a peak, then follows a
//! cosine curve back down. This is the standard recipe used by both
//! I-JEPA and V-JEPA.
//!
//! Two implementations are provided:
//!
//! | Schedule | Behaviour |
//! |----------|-----------|
//! | [`WarmupCosineSchedule`] | Linear warmup → cosine decay (default for JEPA) |
//! | [`ConstantSchedule`] | Fixed LR for testing or fine-tuning |

/// A learning rate schedule.
///
/// Returns the learning rate for a given training step.
pub trait LrSchedule {
    /// Get the learning rate at the given step.
    fn get_lr(&self, step: usize) -> f64;
}

/// Warmup + cosine decay learning rate schedule.
///
/// This is the standard schedule used by I-JEPA and V-JEPA:
/// 1. Linear warmup from `start_lr` to `peak_lr` over `warmup_steps`
/// 2. Cosine decay from `peak_lr` to `end_lr` over the remaining steps
///
/// ```text
///      peak_lr ─────╮
///                    ╲
///  start_lr ╱         ╲
///          │           ╲
///   end_lr │            ╲───
///          ├──────┼──────────┤
///          0   warmup     total
/// ```
///
/// # Example
///
/// ```
/// use jepa_train::schedule::{WarmupCosineSchedule, LrSchedule};
///
/// let schedule = WarmupCosineSchedule::new(1e-3, 1000, 10000);
///
/// // LR starts near zero
/// assert!(schedule.get_lr(0) < 1e-6);
///
/// // LR reaches peak at end of warmup
/// assert!((schedule.get_lr(1000) - 1e-3).abs() < 1e-6);
///
/// // LR decays toward end_lr
/// assert!(schedule.get_lr(10000) < 1e-4);
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WarmupCosineSchedule {
    /// Starting learning rate (at step 0).
    pub start_lr: f64,
    /// Peak learning rate (at end of warmup).
    pub peak_lr: f64,
    /// Final learning rate (at end of training).
    pub end_lr: f64,
    /// Number of warmup steps.
    pub warmup_steps: usize,
    /// Total number of training steps.
    pub total_steps: usize,
}

impl WarmupCosineSchedule {
    /// Create a typical JEPA training schedule.
    ///
    /// Uses warmup from 0 to peak, then cosine decay to near-zero.
    pub fn new(peak_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            start_lr: 0.0,
            peak_lr,
            end_lr: 1e-6,
            warmup_steps,
            total_steps,
        }
    }
}

impl LrSchedule for WarmupCosineSchedule {
    fn get_lr(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            let progress = step as f64 / self.warmup_steps.max(1) as f64;
            self.start_lr + (self.peak_lr - self.start_lr) * progress
        } else {
            // Cosine decay
            let decay_steps = self.total_steps.saturating_sub(self.warmup_steps);
            if decay_steps == 0 {
                return self.end_lr;
            }
            let progress = (step - self.warmup_steps) as f64 / decay_steps as f64;
            let progress = progress.min(1.0);
            self.end_lr
                + (self.peak_lr - self.end_lr) * (1.0 + (progress * std::f64::consts::PI).cos())
                    / 2.0
        }
    }
}

/// Constant learning rate (for baselines and testing).
#[derive(Debug, Clone)]
pub struct ConstantSchedule {
    /// The constant learning rate.
    pub lr: f64,
}

impl LrSchedule for ConstantSchedule {
    fn get_lr(&self, _step: usize) -> f64 {
        self.lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_warmup_cosine_at_start() {
        let schedule = WarmupCosineSchedule::new(1e-3, 1000, 10000);
        let lr = schedule.get_lr(0);
        assert!(lr.abs() < 1e-10, "start lr should be ~0, got {lr}");
    }

    #[test]
    fn test_warmup_cosine_at_warmup_end() {
        let schedule = WarmupCosineSchedule::new(1e-3, 1000, 10000);
        let lr = schedule.get_lr(1000);
        assert!(
            (lr - 1e-3).abs() < 1e-6,
            "lr at warmup end should be peak, got {lr}"
        );
    }

    #[test]
    fn test_warmup_cosine_at_end() {
        let schedule = WarmupCosineSchedule::new(1e-3, 1000, 10000);
        let lr = schedule.get_lr(10000);
        assert!(
            (lr - 1e-6).abs() < 1e-8,
            "lr at end should be ~end_lr, got {lr}"
        );
    }

    #[test]
    fn test_warmup_linear_interpolation() {
        let schedule = WarmupCosineSchedule::new(1e-3, 1000, 10000);
        let lr = schedule.get_lr(500);
        let expected = 5e-4; // half of peak
        assert!(
            (lr - expected).abs() < 1e-6,
            "lr at warmup midpoint should be ~{expected}, got {lr}"
        );
    }

    #[test]
    fn test_warmup_cosine_is_monotonic_during_warmup() {
        let schedule = WarmupCosineSchedule::new(1e-3, 100, 1000);
        let mut prev = schedule.get_lr(0);
        for step in 1..100 {
            let curr = schedule.get_lr(step);
            assert!(
                curr >= prev - 1e-12,
                "lr should increase during warmup: step {step}, {prev} -> {curr}"
            );
            prev = curr;
        }
    }

    #[test]
    fn test_warmup_cosine_is_monotonic_during_decay() {
        let schedule = WarmupCosineSchedule::new(1e-3, 100, 1000);
        let mut prev = schedule.get_lr(100);
        for step in 101..=1000 {
            let curr = schedule.get_lr(step);
            assert!(
                curr <= prev + 1e-12,
                "lr should decrease during decay: step {step}, {prev} -> {curr}"
            );
            prev = curr;
        }
    }

    #[test]
    fn test_constant_schedule() {
        let schedule = ConstantSchedule { lr: 0.01 };
        assert_eq!(schedule.get_lr(0), 0.01);
        assert_eq!(schedule.get_lr(1000), 0.01);
        assert_eq!(schedule.get_lr(999999), 0.01);
    }

    proptest! {
        #[test]
        fn prop_warmup_cosine_lr_always_non_negative(
            step in 0usize..20000,
            peak_lr in 1e-5f64..1e-2,
            warmup in 100usize..5000,
            total in 5000usize..20000,
        ) {
            let schedule = WarmupCosineSchedule::new(peak_lr, warmup.min(total), total);
            let lr = schedule.get_lr(step);
            prop_assert!(lr >= 0.0, "LR was negative: {lr} at step {step}");
            prop_assert!(lr.is_finite(), "LR was not finite at step {step}");
        }

        #[test]
        fn prop_warmup_cosine_lr_bounded_by_peak(
            step in 0usize..20000,
            peak_lr in 1e-5f64..1e-2,
        ) {
            let schedule = WarmupCosineSchedule::new(peak_lr, 1000, 10000);
            let lr = schedule.get_lr(step);
            prop_assert!(
                lr <= peak_lr + 1e-12,
                "LR {lr} exceeded peak {peak_lr} at step {step}"
            );
        }

        #[test]
        fn prop_warmup_is_monotonically_increasing(
            warmup_steps in 10usize..500,
        ) {
            let schedule = WarmupCosineSchedule::new(1e-3, warmup_steps, warmup_steps * 10);
            let mut prev = schedule.get_lr(0);
            for step in 1..warmup_steps {
                let curr = schedule.get_lr(step);
                prop_assert!(
                    curr >= prev - 1e-12,
                    "LR not monotonic during warmup at step {step}: {prev} -> {curr}"
                );
                prev = curr;
            }
        }
    }
}
