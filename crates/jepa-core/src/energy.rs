//! Energy functions for measuring representation compatibility.
//!
//! Implements RFC-004 (Energy Functions).
//!
//! Energy functions measure the distance between predicted and actual
//! target representations. Lower energy = better prediction = more
//! compatible pair.

use burn::tensor::{backend::Backend, Tensor};

use crate::types::{Energy, Representation};

/// Trait for energy functions.
///
/// An energy function measures how compatible two representations are.
/// In JEPA training, this is the primary loss signal: the model learns
/// to minimize the energy between predicted and actual target representations.
///
/// # Example
///
/// ```
/// use jepa_core::energy::{EnergyFn, L2Energy};
/// use jepa_core::types::Representation;
/// use burn::tensor::{Tensor, ElementConversion};
/// use burn_ndarray::NdArray;
///
/// type B = NdArray<f32>;
/// let device = burn_ndarray::NdArrayDevice::Cpu;
///
/// let predicted = Representation::<B>::random([2, 4, 16], &device);
/// let actual = Representation::<B>::random([2, 4, 16], &device);
///
/// let energy = L2Energy.compute(&predicted, &actual);
/// let val: f32 = energy.value.into_scalar().elem();
/// assert!(val >= 0.0);
/// ```
pub trait EnergyFn<B: Backend> {
    /// Compute energy (distance) between predicted and actual representations.
    ///
    /// Lower energy = better prediction = more compatible pair.
    ///
    /// # Arguments
    /// * `predicted` - Representation predicted by the predictor. Shape: `[batch, seq_len, embed_dim]`
    /// * `actual` - Actual target representation from the target encoder. Shape: `[batch, seq_len, embed_dim]`
    fn compute(&self, predicted: &Representation<B>, actual: &Representation<B>) -> Energy<B>;
}

/// L2 distance in representation space (used by I-JEPA, V-JEPA).
///
/// Computes the mean squared error between predicted and actual
/// representations, averaged over all dimensions.
pub struct L2Energy;

impl<B: Backend> EnergyFn<B> for L2Energy {
    fn compute(&self, predicted: &Representation<B>, actual: &Representation<B>) -> Energy<B> {
        let diff = predicted.embeddings.clone() - actual.embeddings.clone();
        let squared = diff.clone() * diff;
        let mean = squared.mean();
        Energy {
            value: mean.unsqueeze(),
        }
    }
}

/// Cosine similarity energy.
///
/// Computes `1 - cosine_similarity` so that identical representations
/// yield energy ≈ 0 and orthogonal representations yield energy ≈ 1.
pub struct CosineEnergy;

impl<B: Backend> EnergyFn<B> for CosineEnergy {
    fn compute(&self, predicted: &Representation<B>, actual: &Representation<B>) -> Energy<B> {
        // Flatten to [batch * seq_len, embed_dim] for per-token cosine similarity
        let [batch, seq_len, embed_dim] = predicted.embeddings.dims();
        let p = predicted
            .embeddings
            .clone()
            .reshape([batch * seq_len, embed_dim]);
        let a = actual
            .embeddings
            .clone()
            .reshape([batch * seq_len, embed_dim]);

        // dot product along embed_dim
        let dot = (p.clone() * a.clone()).sum_dim(1);

        // norms
        let norm_p = (p.clone() * p).sum_dim(1).sqrt();
        let norm_a = (a.clone() * a).sum_dim(1).sqrt();

        // cosine similarity with eps for numerical stability
        let eps: f64 = 1e-8;
        let cos_sim = dot / (norm_p * norm_a + eps);

        // energy = 1 - mean(cos_sim)
        let one: Tensor<B, 1> = Tensor::ones([1], &cos_sim.device());
        let energy_value = one - cos_sim.mean().unsqueeze();

        Energy {
            value: energy_value,
        }
    }
}

/// Smooth L1 energy (Huber loss variant).
///
/// Behaves as L2 for small differences (< beta) and L1 for large differences.
/// This makes it less sensitive to outliers than pure L2.
pub struct SmoothL1Energy {
    /// Threshold below which the loss is L2-like.
    pub beta: f64,
}

impl SmoothL1Energy {
    /// Create a new SmoothL1Energy with the given beta threshold.
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }
}

impl<B: Backend> EnergyFn<B> for SmoothL1Energy {
    fn compute(&self, predicted: &Representation<B>, actual: &Representation<B>) -> Energy<B> {
        let diff = predicted.embeddings.clone() - actual.embeddings.clone();
        let abs_diff = diff.abs();

        let beta_tensor: Tensor<B, 3> =
            Tensor::full(abs_diff.dims(), self.beta, &abs_diff.device());

        // Where |diff| < beta: 0.5 * diff^2 / beta
        // Where |diff| >= beta: |diff| - 0.5 * beta
        let quadratic = abs_diff.clone() * abs_diff.clone() * 0.5 / self.beta;
        let linear = abs_diff.clone() - 0.5 * self.beta;

        // mask: 1.0 where abs_diff < beta, 0.0 otherwise
        let mask = abs_diff.lower(beta_tensor).float();
        let one_minus_mask = mask.clone().neg() + 1.0;

        let loss = quadratic * mask + linear * one_minus_mask;
        let mean = loss.mean();

        Energy {
            value: mean.unsqueeze(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::ElementConversion;
    use burn_ndarray::NdArray;
    use proptest::prelude::*;
    use rand::Rng as _;
    use rand::SeedableRng;

    type TestBackend = NdArray<f32>;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

    fn make_repr(data: &[f32], shape: [usize; 3]) -> Representation<TestBackend> {
        Representation::new(Tensor::from_floats(
            burn::tensor::TensorData::new(data.to_vec(), shape),
            &device(),
        ))
    }

    #[test]
    fn test_l2_energy_identical_representations_is_zero() {
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let repr = make_repr(&data, [2, 3, 4]);
        let energy = L2Energy.compute(&repr, &repr);
        let val: f32 = energy.value.into_scalar().elem();
        assert!(val.abs() < 1e-6, "expected ~0, got {val}");
    }

    #[test]
    fn test_l2_energy_different_representations_is_positive() {
        let a_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..24).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let a = make_repr(&a_data, [2, 3, 4]);
        let b = make_repr(&b_data, [2, 3, 4]);
        let energy = L2Energy.compute(&a, &b);
        let val: f32 = energy.value.into_scalar().elem();
        assert!(val > 0.0, "expected positive, got {val}");
    }

    #[test]
    fn test_l2_energy_is_symmetric() {
        let a_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..24).map(|i| (i as f32 + 5.0) * 0.3).collect();
        let a = make_repr(&a_data, [2, 3, 4]);
        let b = make_repr(&b_data, [2, 3, 4]);
        let e_ab: f32 = L2Energy.compute(&a, &b).value.into_scalar().elem();
        let e_ba: f32 = L2Energy.compute(&b, &a).value.into_scalar().elem();
        assert!(
            (e_ab - e_ba).abs() < 1e-6,
            "L2 energy not symmetric: {e_ab} vs {e_ba}"
        );
    }

    #[test]
    fn test_cosine_energy_identical_is_near_zero() {
        let data: Vec<f32> = (1..25).map(|i| i as f32).collect();
        let repr = make_repr(&data, [2, 3, 4]);
        let energy = CosineEnergy.compute(&repr, &repr);
        let val: f32 = energy.value.into_scalar().elem();
        assert!(val.abs() < 1e-5, "expected ~0, got {val}");
    }

    #[test]
    fn test_cosine_energy_orthogonal_is_near_one() {
        // Two 2D vectors that are orthogonal: [1, 0] and [0, 1]
        let a = make_repr(&[1.0, 0.0], [1, 1, 2]);
        let b = make_repr(&[0.0, 1.0], [1, 1, 2]);
        let energy = CosineEnergy.compute(&a, &b);
        let val: f32 = energy.value.into_scalar().elem();
        assert!(
            (val - 1.0).abs() < 1e-5,
            "expected ~1.0 for orthogonal, got {val}"
        );
    }

    #[test]
    fn test_smooth_l1_identical_is_zero() {
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let repr = make_repr(&data, [1, 3, 4]);
        let energy = SmoothL1Energy::new(1.0).compute(&repr, &repr);
        let val: f32 = energy.value.into_scalar().elem();
        assert!(val.abs() < 1e-6, "expected ~0, got {val}");
    }

    #[test]
    fn test_smooth_l1_is_non_negative() {
        let a_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
        let b_data: Vec<f32> = (0..12).map(|i| (i as f32 + 2.0) * 0.3).collect();
        let a = make_repr(&a_data, [1, 3, 4]);
        let b = make_repr(&b_data, [1, 3, 4]);
        let energy = SmoothL1Energy::new(1.0).compute(&a, &b);
        let val: f32 = energy.value.into_scalar().elem();
        assert!(val >= 0.0, "expected non-negative, got {val}");
    }

    // --- Property-based tests ---

    proptest! {
        #[test]
        fn prop_l2_energy_never_negative(seed in 0u64..10000) {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            let a_data: Vec<f32> = (0..24).map(|_| (rng.gen::<f32>() - 0.5) * 10.0).collect();
            let b_data: Vec<f32> = (0..24).map(|_| (rng.gen::<f32>() - 0.5) * 10.0).collect();
            let a = make_repr(&a_data, [2, 3, 4]);
            let b = make_repr(&b_data, [2, 3, 4]);
            let val: f32 = L2Energy.compute(&a, &b).value.into_scalar().elem();
            prop_assert!(val >= 0.0, "L2 energy was negative: {val}");
            prop_assert!(val.is_finite(), "L2 energy was not finite: {val}");
        }

        #[test]
        fn prop_l2_energy_is_symmetric(seed in 0u64..10000) {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            let a_data: Vec<f32> = (0..24).map(|_| rng.gen::<f32>() * 5.0).collect();
            let b_data: Vec<f32> = (0..24).map(|_| rng.gen::<f32>() * 5.0).collect();
            let a = make_repr(&a_data, [2, 3, 4]);
            let b = make_repr(&b_data, [2, 3, 4]);
            let e_ab: f32 = L2Energy.compute(&a, &b).value.into_scalar().elem();
            let e_ba: f32 = L2Energy.compute(&b, &a).value.into_scalar().elem();
            prop_assert!((e_ab - e_ba).abs() < 1e-5, "not symmetric: {e_ab} vs {e_ba}");
        }

        #[test]
        fn prop_l2_energy_zero_for_identical(seed in 0u64..10000) {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            let data: Vec<f32> = (0..24).map(|_| rng.gen::<f32>() * 10.0).collect();
            let repr = make_repr(&data, [2, 3, 4]);
            let val: f32 = L2Energy.compute(&repr, &repr).value.into_scalar().elem();
            prop_assert!(val.abs() < 1e-6, "expected ~0 for identical, got {val}");
        }

        #[test]
        fn prop_smooth_l1_never_negative(seed in 0u64..10000) {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            let a_data: Vec<f32> = (0..12).map(|_| rng.gen::<f32>() * 10.0).collect();
            let b_data: Vec<f32> = (0..12).map(|_| rng.gen::<f32>() * 10.0).collect();
            let a = make_repr(&a_data, [1, 3, 4]);
            let b = make_repr(&b_data, [1, 3, 4]);
            let val: f32 = SmoothL1Energy::new(1.0).compute(&a, &b).value.into_scalar().elem();
            prop_assert!(val >= 0.0, "SmoothL1 was negative: {val}");
        }
    }
}
