//! Exponential Moving Average (EMA) for target encoder updates.
//!
//! Implements RFC-007 (EMA Target Encoder).
//!
//! In JEPA, the target encoder is not trained via gradient descent.
//! Instead, its weights are an EMA of the context encoder's weights.
//! This asymmetry is critical for preventing collapse.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Exponential Moving Average weight updater.
///
/// Updates target parameters toward online (context) parameters:
/// ```text
/// target = momentum * target + (1 - momentum) * online
/// ```
///
/// Higher momentum means the target changes more slowly,
/// providing a more stable prediction target.
///
/// # Example
///
/// ```
/// use jepa_core::ema::Ema;
/// use burn::tensor::Tensor;
/// use burn_ndarray::NdArray;
///
/// type B = NdArray<f32>;
/// let device = burn_ndarray::NdArrayDevice::Cpu;
///
/// // Constant momentum
/// let ema = Ema::new(0.996);
/// let target: Tensor<B, 1> = Tensor::zeros([8], &device);
/// let online: Tensor<B, 1> = Tensor::ones([8], &device);
/// let updated = ema.update_tensor(target, &online, 0);
///
/// // With cosine schedule (momentum increases over training)
/// let ema_scheduled = Ema::with_cosine_schedule(0.996, 10000);
/// assert!((ema_scheduled.get_momentum(0) - 0.996).abs() < 1e-6);
/// assert!((ema_scheduled.get_momentum(9999) - 1.0).abs() < 1e-3);
/// ```
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

    /// Perform an EMA update on a pair of tensors.
    ///
    /// Computes: `target = momentum * target + (1 - momentum) * online`
    ///
    /// This is the core operation for updating the target encoder's parameters
    /// from the context encoder's parameters during JEPA training.
    ///
    /// # Arguments
    /// * `target` - The target tensor to update (e.g., target encoder weight)
    /// * `online` - The online tensor (e.g., context encoder weight)
    /// * `step` - Current training step (used to compute scheduled momentum)
    ///
    /// # Returns
    /// The updated target tensor
    pub fn update_tensor<B: Backend, const D: usize>(
        &self,
        target: Tensor<B, D>,
        online: &Tensor<B, D>,
        step: usize,
    ) -> Tensor<B, D> {
        let m = self.get_momentum(step);
        target * m + online.clone() * (1.0 - m)
    }

    /// Perform an EMA update on a list of parameter tensor pairs.
    ///
    /// Updates each target parameter tensor in place using the EMA formula.
    /// This is designed for updating all parameters of a target encoder
    /// from a context encoder in a single call.
    ///
    /// # Arguments
    /// * `pairs` - Iterator of (target, online) tensor pairs
    /// * `step` - Current training step
    ///
    /// # Returns
    /// The updated target tensors
    pub fn update_tensor_pairs<B: Backend, const D: usize>(
        &self,
        pairs: Vec<(Tensor<B, D>, Tensor<B, D>)>,
        step: usize,
    ) -> Vec<Tensor<B, D>> {
        let m = self.get_momentum(step);
        pairs
            .into_iter()
            .map(|(target, online)| target * m + online * (1.0 - m))
            .collect()
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
    use burn_ndarray::NdArray;
    use proptest::prelude::*;

    type TestBackend = NdArray<f32>;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

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

    // --- Tensor-level EMA tests ---

    #[test]
    fn test_tensor_ema_momentum_1_keeps_target() {
        let ema = Ema::new(1.0);
        let target: Tensor<TestBackend, 2> =
            Tensor::from_floats([[1.0, 2.0], [3.0, 4.0]], &device());
        let online: Tensor<TestBackend, 2> =
            Tensor::from_floats([[10.0, 20.0], [30.0, 40.0]], &device());

        let result = ema.update_tensor(target, &online, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_ema_momentum_0_copies_online() {
        let ema = Ema::new(0.0);
        let target: Tensor<TestBackend, 2> =
            Tensor::from_floats([[1.0, 2.0], [3.0, 4.0]], &device());
        let online: Tensor<TestBackend, 2> =
            Tensor::from_floats([[10.0, 20.0], [30.0, 40.0]], &device());

        let result = ema.update_tensor(target, &online, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert!((data[0] - 10.0).abs() < 1e-6);
        assert!((data[3] - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_ema_typical_momentum() {
        let ema = Ema::new(0.996);
        let target: Tensor<TestBackend, 1> = Tensor::zeros([4], &device());
        let online: Tensor<TestBackend, 1> = Tensor::ones([4], &device());

        let result = ema.update_tensor(target, &online, 0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        // 0.996 * 0.0 + 0.004 * 1.0 = 0.004
        for &v in &data {
            assert!((v - 0.004).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tensor_ema_convergence() {
        let ema = Ema::new(0.99);
        let online: Tensor<TestBackend, 1> = Tensor::ones([8], &device());
        let mut target: Tensor<TestBackend, 1> = Tensor::zeros([8], &device());

        for step in 0..1000 {
            target = ema.update_tensor(target, &online, step);
        }

        let data: Vec<f32> = target.into_data().to_vec().unwrap();
        for &v in &data {
            assert!(
                (v - 1.0).abs() < 0.01,
                "expected convergence to 1.0, got {v}"
            );
        }
    }

    #[test]
    fn test_tensor_ema_with_schedule() {
        let ema = Ema::with_cosine_schedule(0.996, 100);
        let target: Tensor<TestBackend, 1> = Tensor::zeros([4], &device());
        let online: Tensor<TestBackend, 1> = Tensor::ones([4], &device());

        // Early step: low momentum (moves more toward online)
        let result_early = ema.update_tensor(target.clone(), &online, 0);
        let early: Vec<f32> = result_early.into_data().to_vec().unwrap();

        // Late step: high momentum (moves less toward online)
        let result_late = ema.update_tensor(target, &online, 99);
        let late: Vec<f32> = result_late.into_data().to_vec().unwrap();

        // Early should be further from 0 (closer to online) than late
        assert!(
            early[0] > late[0],
            "early step ({}) should move more than late step ({})",
            early[0],
            late[0]
        );
    }

    #[test]
    fn test_tensor_pair_update() {
        let ema = Ema::new(0.5);
        let pairs = vec![
            (
                Tensor::<TestBackend, 1>::zeros([4], &device()),
                Tensor::<TestBackend, 1>::ones([4], &device()),
            ),
            (
                Tensor::<TestBackend, 1>::ones([4], &device()),
                Tensor::<TestBackend, 1>::zeros([4], &device()),
            ),
        ];

        let results = ema.update_tensor_pairs(pairs, 0);
        assert_eq!(results.len(), 2);

        // First pair: 0.5 * 0 + 0.5 * 1 = 0.5
        let d0: Vec<f32> = results[0].clone().into_data().to_vec().unwrap();
        assert!((d0[0] - 0.5).abs() < 1e-6);

        // Second pair: 0.5 * 1 + 0.5 * 0 = 0.5
        let d1: Vec<f32> = results[1].clone().into_data().to_vec().unwrap();
        assert!((d1[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_ema_3d_shape_preserved() {
        let ema = Ema::new(0.99);
        let target: Tensor<TestBackend, 3> = Tensor::zeros([2, 4, 8], &device());
        let online: Tensor<TestBackend, 3> = Tensor::ones([2, 4, 8], &device());

        let result = ema.update_tensor(target, &online, 0);
        assert_eq!(result.dims(), [2, 4, 8]);
    }

    // --- Property-based tests ---

    proptest! {
        #[test]
        fn prop_ema_converges_to_online(
            momentum in 0.9f64..0.995,
            steps in 1000usize..10000,
        ) {
            let ema = Ema::new(momentum);
            let online = 1.0f64;
            let mut target = 0.0f64;

            for s in 0..steps {
                target = ema.step(target, online, s);
            }

            // After many steps, target should be close to online
            prop_assert!(
                (target - online).abs() < 0.1,
                "did not converge: momentum={momentum}, steps={steps}, target={target}"
            );
        }

        #[test]
        fn prop_ema_momentum_bounds(
            momentum in 0.0f64..=1.0f64,
            target_val in -100.0f64..100.0,
            online_val in -100.0f64..100.0,
        ) {
            let ema = Ema::new(momentum);
            let result = ema.step(target_val, online_val, 0);

            // Result should be between target and online (convex combination)
            let lo = target_val.min(online_val);
            let hi = target_val.max(online_val);
            prop_assert!(
                result >= lo - 1e-10 && result <= hi + 1e-10,
                "result {result} out of bounds [{lo}, {hi}] with momentum {momentum}"
            );
        }

        #[test]
        fn prop_tensor_ema_matches_scalar(
            momentum in 0.5f64..0.999,
        ) {
            let ema = Ema::new(momentum);

            let target_val = 3.0f32;
            let online_val = 7.0f32;

            let scalar_result = ema.step(target_val as f64, online_val as f64, 0) as f32;

            let target: Tensor<TestBackend, 1> = Tensor::from_floats([target_val], &device());
            let online: Tensor<TestBackend, 1> = Tensor::from_floats([online_val], &device());
            let tensor_result: Vec<f32> = ema.update_tensor(target, &online, 0)
                .into_data().to_vec().unwrap();

            prop_assert!(
                (tensor_result[0] - scalar_result).abs() < 1e-4,
                "scalar={scalar_result}, tensor={}", tensor_result[0]
            );
        }
    }
}
