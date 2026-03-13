//! Collapse prevention regularizers for JEPA.
//!
//! Implements RFC-006 (Collapse Prevention).
//!
//! Without explicit regularization, joint-embedding models can
//! *collapse* — all inputs map to the same constant representation,
//! yielding zero energy but zero information. While EMA target updates
//! (see [`crate::ema`]) help, they are not sufficient alone. This
//! module provides two complementary regularizers:
//!
//! | Regularizer | Strategy | Reference |
//! |-------------|----------|-----------|
//! | [`VICReg`] | Variance + Invariance + Covariance | Bardes et al. (2022), VICReg |
//! | [`BarlowTwins`] | Cross-correlation → identity | Zbontar et al. (2021), Barlow Twins |
//!
//! Both implement the [`CollapseRegularizer`] trait and return a scalar
//! loss term that should be added (with appropriate weighting) to the
//! energy loss during training.

use burn::tensor::{backend::Backend, Tensor};

/// Trait for collapse prevention regularizers.
///
/// Collapse regularizers add a loss term that prevents all representations
/// from converging to the same point in embedding space.
pub trait CollapseRegularizer<B: Backend> {
    /// Compute the regularization loss.
    ///
    /// # Arguments
    /// * `z_a` - First set of representations. Shape: `[batch, embed_dim]`
    /// * `z_b` - Second set of representations. Shape: `[batch, embed_dim]`
    ///
    /// # Returns
    /// A scalar loss tensor (higher means more collapse detected).
    fn loss(&self, z_a: &Tensor<B, 2>, z_b: &Tensor<B, 2>) -> Tensor<B, 1>;
}

/// VICReg regularization loss components.
///
/// Contains the three VICReg terms for inspection and logging.
#[derive(Debug, Clone)]
pub struct VICRegLoss<B: Backend> {
    /// Invariance: MSE between paired representations.
    pub invariance: Tensor<B, 1>,
    /// Variance: hinge loss on per-dimension standard deviation.
    pub variance: Tensor<B, 1>,
    /// Covariance: penalty on off-diagonal covariance elements.
    pub covariance: Tensor<B, 1>,
}

impl<B: Backend> VICRegLoss<B> {
    /// Total weighted loss: invariance + variance + covariance.
    pub fn total(&self) -> Tensor<B, 1> {
        self.invariance.clone() + self.variance.clone() + self.covariance.clone()
    }
}

/// VICReg (Variance-Invariance-Covariance Regularization).
///
/// Prevents collapse by enforcing:
/// 1. **Variance**: each embedding dimension has high variance across the batch
/// 2. **Invariance**: representations of positive pairs are similar
/// 3. **Covariance**: different embedding dimensions capture different information
///
/// # Example
///
/// ```
/// use jepa_core::collapse::{VICReg, CollapseRegularizer};
/// use burn::tensor::{Tensor, ElementConversion};
/// use burn_ndarray::NdArray;
///
/// type B = NdArray<f32>;
/// let device = burn_ndarray::NdArrayDevice::Cpu;
///
/// let vicreg = VICReg::default();
/// let z_a: Tensor<B, 2> = Tensor::random(
///     [32, 64],
///     burn::tensor::Distribution::Normal(0.0, 1.0),
///     &device,
/// );
/// let z_b = z_a.clone();
///
/// // Compute decomposed loss for logging
/// let loss = vicreg.compute(&z_a, &z_b);
/// let total: f32 = loss.total().into_scalar().elem();
/// assert!(total.is_finite());
///
/// // Or use the trait interface for a single scalar
/// let scalar_loss: f32 = vicreg.loss(&z_a, &z_b).into_scalar().elem();
/// assert!(scalar_loss.is_finite());
/// ```
pub struct VICReg {
    /// Weight for the invariance term (default: 25.0).
    pub lambda_inv: f64,
    /// Weight for the variance term (default: 25.0).
    pub mu_var: f64,
    /// Weight for the covariance term (default: 1.0).
    pub nu_cov: f64,
    /// Target standard deviation for variance hinge loss (default: 1.0).
    pub gamma: f64,
    /// Epsilon for numerical stability (default: 1e-4).
    pub eps: f64,
}

impl Default for VICReg {
    fn default() -> Self {
        Self {
            lambda_inv: 25.0,
            mu_var: 25.0,
            nu_cov: 1.0,
            gamma: 1.0,
            eps: 1e-4,
        }
    }
}

impl VICReg {
    /// Compute the decomposed VICReg loss.
    ///
    /// Returns individual variance, invariance, and covariance terms
    /// for inspection. Use [`VICRegLoss::total`] for the combined loss.
    ///
    /// # Arguments
    /// * `z_a` - Representations from branch A. Shape: `[batch, embed_dim]`
    /// * `z_b` - Representations from branch B. Shape: `[batch, embed_dim]`
    pub fn compute<B: Backend>(&self, z_a: &Tensor<B, 2>, z_b: &Tensor<B, 2>) -> VICRegLoss<B> {
        let device = z_a.device();
        let [batch, embed_dim] = z_a.dims();
        let n = batch as f64;

        // --- Invariance: MSE between paired representations ---
        let diff = z_a.clone() - z_b.clone();
        let inv_loss = (diff.clone() * diff).mean().reshape([1]);

        // --- Variance: hinge loss on std dev ---
        let var_loss = self.variance_loss(z_a, n, &device) + self.variance_loss(z_b, n, &device);

        // --- Covariance: off-diagonal elements should be zero ---
        let cov_loss =
            self.covariance_loss(z_a, n, embed_dim) + self.covariance_loss(z_b, n, embed_dim);

        VICRegLoss {
            invariance: inv_loss * self.lambda_inv,
            variance: var_loss * self.mu_var,
            covariance: cov_loss * self.nu_cov,
        }
    }

    fn variance_loss<B: Backend>(
        &self,
        z: &Tensor<B, 2>,
        n: f64,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        // Compute per-dimension variance: var = E[(x - mean)^2]
        let mean = z.clone().mean_dim(0); // [1, embed_dim]
        let centered = z.clone() - mean;
        let var = (centered.clone() * centered).sum_dim(0) / (n - 1.0).max(1.0); // [1, embed_dim]
        let std = (var + self.eps).sqrt();

        // Hinge: max(0, gamma - std)
        let gamma_tensor: Tensor<B, 2> = Tensor::full(std.dims(), self.gamma, device);
        let hinge = (gamma_tensor - std).clamp_min(0.0);
        // mean() on [1, embed_dim] → reshape to [1]
        hinge.mean().reshape([1])
    }

    fn covariance_loss<B: Backend>(
        &self,
        z: &Tensor<B, 2>,
        n: f64,
        embed_dim: usize,
    ) -> Tensor<B, 1> {
        // Center the representations
        let mean = z.clone().mean_dim(0); // [1, embed_dim]
        let centered = z.clone() - mean; // [batch, embed_dim]

        // Covariance matrix: C = (1/N) * Z^T Z
        let cov = centered.clone().transpose().matmul(centered) / (n - 1.0).max(1.0); // [embed_dim, embed_dim]

        // Zero out diagonal (we only penalize off-diagonal)
        let diag_mask = Tensor::eye(embed_dim, &cov.device()); // [embed_dim, embed_dim]
        let off_diag = cov * (diag_mask.neg() + 1.0);

        // Sum of squared off-diagonal elements, normalized → reshape to [1]
        ((off_diag.clone() * off_diag).sum() / embed_dim as f64).reshape([1])
    }
}

impl<B: Backend> CollapseRegularizer<B> for VICReg {
    fn loss(&self, z_a: &Tensor<B, 2>, z_b: &Tensor<B, 2>) -> Tensor<B, 1> {
        self.compute(z_a, z_b).total()
    }
}

/// Barlow Twins regularization.
///
/// Prevents collapse by making the cross-correlation matrix between two
/// representation branches approach the identity matrix. This enforces:
/// 1. **Invariance**: diagonal elements ≈ 1 (paired representations are similar)
/// 2. **Redundancy reduction**: off-diagonal elements ≈ 0 (dimensions are decorrelated)
///
/// Reference: Zbontar et al. (2021), "Barlow Twins: Self-Supervised Learning via
/// Redundancy Reduction", ICML.
pub struct BarlowTwins {
    /// Weight for the off-diagonal (redundancy reduction) term (default: 0.005).
    ///
    /// The on-diagonal (invariance) term always has weight 1.0.
    /// Lower lambda means less penalty on redundant features.
    pub lambda_bt: f64,
}

impl Default for BarlowTwins {
    fn default() -> Self {
        Self { lambda_bt: 0.005 }
    }
}

/// Decomposed Barlow Twins loss components.
#[derive(Debug, Clone)]
pub struct BarlowTwinsLoss<B: Backend> {
    /// On-diagonal loss: `sum((1 - diag(C))^2)`. Shape: `[1]`
    pub on_diagonal: Tensor<B, 1>,
    /// Off-diagonal loss: `sum(off_diag(C)^2)`. Shape: `[1]`
    pub off_diagonal: Tensor<B, 1>,
}

impl<B: Backend> BarlowTwinsLoss<B> {
    /// Total loss: on_diagonal + lambda * off_diagonal.
    pub fn total(&self) -> Tensor<B, 1> {
        self.on_diagonal.clone() + self.off_diagonal.clone()
    }
}

impl BarlowTwins {
    /// Create a new Barlow Twins regularizer with the given lambda.
    pub fn new(lambda_bt: f64) -> Self {
        Self { lambda_bt }
    }

    /// Compute the decomposed Barlow Twins loss.
    ///
    /// 1. Standardize each branch (zero mean, unit std per dimension)
    /// 2. Compute the cross-correlation matrix C = (1/N) * Z_a^T * Z_b
    /// 3. On-diagonal: penalize deviation from 1 (invariance)
    /// 4. Off-diagonal: penalize non-zero elements (redundancy reduction)
    ///
    /// # Arguments
    /// * `z_a` - Representations from branch A. Shape: `[batch, embed_dim]`
    /// * `z_b` - Representations from branch B. Shape: `[batch, embed_dim]`
    pub fn compute<B: Backend>(
        &self,
        z_a: &Tensor<B, 2>,
        z_b: &Tensor<B, 2>,
    ) -> BarlowTwinsLoss<B> {
        let [batch, embed_dim] = z_a.dims();
        let n = batch as f64;
        let eps = 1e-5;

        // Standardize each branch: z_norm = (z - mean) / std
        let z_a_norm = standardize::<B>(z_a, n, eps);
        let z_b_norm = standardize::<B>(z_b, n, eps);

        // Cross-correlation matrix: C = (1/N) * Z_a^T * Z_b
        // Shape: [embed_dim, embed_dim]
        let cross_corr = z_a_norm.transpose().matmul(z_b_norm) / n;

        // On-diagonal: sum of (1 - C_ii)^2
        let diag_mask = Tensor::<B, 2>::eye(embed_dim, &cross_corr.device());
        let diag_values = cross_corr.clone() * diag_mask.clone();
        let on_diag_diff = diag_values - diag_mask;
        let on_diag_loss = (on_diag_diff.clone() * on_diag_diff).sum().reshape([1]);

        // Off-diagonal: sum of C_ij^2 for i != j
        let off_diag_mask = Tensor::<B, 2>::eye(embed_dim, &cross_corr.device()).neg() + 1.0;
        let off_diag = cross_corr * off_diag_mask;
        let off_diag_loss = ((off_diag.clone() * off_diag).sum() * self.lambda_bt).reshape([1]);

        BarlowTwinsLoss {
            on_diagonal: on_diag_loss,
            off_diagonal: off_diag_loss,
        }
    }
}

/// Standardize a batch of representations to zero mean, unit std per dimension.
fn standardize<B: Backend>(z: &Tensor<B, 2>, n: f64, eps: f64) -> Tensor<B, 2> {
    let mean = z.clone().mean_dim(0); // [1, embed_dim]
    let centered = z.clone() - mean;
    let var = (centered.clone() * centered.clone()).sum_dim(0) / (n - 1.0).max(1.0);
    let std = (var + eps).sqrt();
    centered / std
}

impl<B: Backend> CollapseRegularizer<B> for BarlowTwins {
    fn loss(&self, z_a: &Tensor<B, 2>, z_b: &Tensor<B, 2>) -> Tensor<B, 1> {
        self.compute(z_a, z_b).total()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{ElementConversion, Tensor};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

    #[test]
    fn test_vicreg_default_params() {
        let vicreg = VICReg::default();
        assert_eq!(vicreg.lambda_inv, 25.0);
        assert_eq!(vicreg.mu_var, 25.0);
        assert_eq!(vicreg.nu_cov, 1.0);
    }

    #[test]
    fn test_collapsed_representations_high_variance_loss() {
        let vicreg = VICReg::default();
        // All-zeros = fully collapsed
        let z: Tensor<TestBackend, 2> = Tensor::zeros([32, 64], &device());
        let loss = vicreg.compute(&z, &z);
        let var_val: f32 = loss.variance.into_scalar().elem();
        // Variance term should be high for collapsed representations
        assert!(
            var_val > 10.0,
            "expected high variance loss for collapse, got {var_val}"
        );
    }

    #[test]
    fn test_identical_pairs_zero_invariance() {
        let vicreg = VICReg::default();
        // Random non-collapsed representations
        let z: Tensor<TestBackend, 2> = Tensor::random(
            [32, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let loss = vicreg.compute(&z, &z);
        let inv_val: f32 = loss.invariance.into_scalar().elem();
        assert!(
            inv_val.abs() < 1e-5,
            "expected ~0 invariance for identical pairs, got {inv_val}"
        );
    }

    #[test]
    fn test_diverse_representations_low_variance_loss() {
        let vicreg = VICReg {
            gamma: 1.0,
            ..VICReg::default()
        };
        // Representations with std dev > 1.0 per dimension
        let z: Tensor<TestBackend, 2> = Tensor::random(
            [32, 64],
            burn::tensor::Distribution::Normal(0.0, 2.0),
            &device(),
        );
        let loss = vicreg.compute(&z, &z.clone());
        let var_val: f32 = loss.variance.into_scalar().elem();
        // With std > gamma, hinge should be near zero
        assert!(
            var_val < 5.0,
            "expected low variance loss for diverse representations, got {var_val}"
        );
    }

    #[test]
    fn test_vicreg_total_loss() {
        let vicreg = VICReg::default();
        let z: Tensor<TestBackend, 2> = Tensor::random(
            [16, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let loss = vicreg.compute(&z, &z);
        let total: f32 = loss.total().into_scalar().elem();
        let sum: f32 = {
            let i: f32 = loss.invariance.into_scalar().elem();
            let v: f32 = loss.variance.into_scalar().elem();
            let c: f32 = loss.covariance.into_scalar().elem();
            i + v + c
        };
        assert!(
            (total - sum).abs() < 1e-4,
            "total() should equal sum of components: {total} vs {sum}"
        );
    }

    #[test]
    fn test_collapse_regularizer_trait() {
        let vicreg = VICReg::default();
        let z: Tensor<TestBackend, 2> = Tensor::random(
            [16, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let loss: Tensor<TestBackend, 1> = vicreg.loss(&z, &z);
        let val: f32 = loss.into_scalar().elem();
        assert!(val.is_finite(), "loss should be finite, got {val}");
    }

    // --- Barlow Twins tests ---

    #[test]
    fn test_barlow_twins_default_params() {
        let bt = BarlowTwins::default();
        assert!((bt.lambda_bt - 0.005).abs() < 1e-10);
    }

    #[test]
    fn test_barlow_twins_identical_inputs_low_on_diagonal_loss() {
        // Identical inputs → cross-correlation diagonal ≈ 1 → on_diagonal loss ≈ 0
        let bt = BarlowTwins::default();
        let z: Tensor<TestBackend, 2> = Tensor::random(
            [64, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let loss = bt.compute(&z, &z);
        let on_diag: f32 = loss.on_diagonal.into_scalar().elem();
        assert!(
            on_diag < 1.0,
            "identical inputs should have low on-diagonal loss, got {on_diag}"
        );
    }

    #[test]
    fn test_barlow_twins_loss_is_finite() {
        let bt = BarlowTwins::default();
        let z_a: Tensor<TestBackend, 2> = Tensor::random(
            [32, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let z_b: Tensor<TestBackend, 2> = Tensor::random(
            [32, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let loss = bt.compute(&z_a, &z_b);
        let total: f32 = loss.total().into_scalar().elem();
        assert!(
            total.is_finite(),
            "total loss should be finite, got {total}"
        );
    }

    #[test]
    fn test_barlow_twins_loss_is_non_negative() {
        let bt = BarlowTwins::default();
        let z: Tensor<TestBackend, 2> = Tensor::random(
            [32, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let loss = bt.compute(&z, &z);
        let on_diag: f32 = loss.on_diagonal.into_scalar().elem();
        let off_diag: f32 = loss.off_diagonal.into_scalar().elem();
        assert!(
            on_diag >= -1e-6,
            "on_diagonal should be >= 0, got {on_diag}"
        );
        assert!(
            off_diag >= -1e-6,
            "off_diagonal should be >= 0, got {off_diag}"
        );
    }

    #[test]
    fn test_barlow_twins_total_equals_sum_of_components() {
        let bt = BarlowTwins::default();
        let z: Tensor<TestBackend, 2> = Tensor::random(
            [32, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let loss = bt.compute(&z, &z);
        let total: f32 = loss.total().into_scalar().elem();
        let on_diag: f32 = loss.on_diagonal.into_scalar().elem();
        let off_diag: f32 = loss.off_diagonal.into_scalar().elem();
        assert!(
            (total - (on_diag + off_diag)).abs() < 1e-4,
            "total should equal on_diag + off_diag: {total} vs {} + {}",
            on_diag,
            off_diag,
        );
    }

    #[test]
    fn test_barlow_twins_implements_collapse_regularizer() {
        let bt = BarlowTwins::default();
        let z: Tensor<TestBackend, 2> = Tensor::random(
            [16, 32],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let loss: Tensor<TestBackend, 1> = CollapseRegularizer::loss(&bt, &z, &z);
        let val: f32 = loss.into_scalar().elem();
        assert!(
            val.is_finite(),
            "Barlow Twins loss should be finite, got {val}"
        );
    }

    #[test]
    fn test_barlow_twins_higher_lambda_increases_off_diagonal_penalty() {
        let z_a: Tensor<TestBackend, 2> = Tensor::random(
            [32, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );
        let z_b: Tensor<TestBackend, 2> = Tensor::random(
            [32, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device(),
        );

        let bt_low = BarlowTwins::new(0.001);
        let bt_high = BarlowTwins::new(0.1);

        let loss_low: f32 = bt_low.compute(&z_a, &z_b).off_diagonal.into_scalar().elem();
        let loss_high: f32 = bt_high
            .compute(&z_a, &z_b)
            .off_diagonal
            .into_scalar()
            .elem();

        assert!(
            loss_high > loss_low,
            "higher lambda should increase off-diagonal penalty: {loss_high} vs {loss_low}"
        );
    }
}
