//! Core data types for JEPA.
//!
//! This module provides the semantic tensor wrappers that form the common
//! vocabulary across every JEPA crate:
//!
//! | Type | Shape | Role |
//! |------|-------|------|
//! | [`Representation`] | `[B, S, D]` | Encoder / predictor output with optional validity mask |
//! | [`Energy`] | `[1]` | Scalar compatibility score (lower = better) |
//! | [`MaskSpec`] | — | Disjoint context / target index partition |
//! | [`InputShape`] | — | 2-D image or 3-D video grid dimensions |
//! | [`MaskError`] | — | Validation errors for [`MaskSpec`] |
//!
//! All tensor-bearing types are generic over `B: Backend`.

use burn::tensor::{backend::Backend, Int, Tensor, TensorData};

/// A representation produced by an encoder.
///
/// Wraps a tensor with semantic meaning. A representation is the compressed,
/// abstract form of an input that captures essential features while discarding
/// irrelevant details.
///
/// Shape: `[batch_size, sequence_length, embed_dim]`
///
/// Per RFC-001, an optional mask tensor indicates which positions are valid,
/// enabling variable-length sequences and masked operations.
#[derive(Debug, Clone)]
pub struct Representation<B: Backend> {
    /// The embedding tensor. Shape: `[batch, seq_len, embed_dim]`
    pub embeddings: Tensor<B, 3>,
    /// Optional mask indicating which positions are valid.
    /// Shape: `[batch, seq_len]`. A value of 1.0 means valid, 0.0 means padding.
    pub mask: Option<Tensor<B, 2>>,
}

impl<B: Backend> Representation<B> {
    /// Create a new representation from a tensor (no mask).
    pub fn new(embeddings: Tensor<B, 3>) -> Self {
        Self {
            embeddings,
            mask: None,
        }
    }

    /// Create a new representation with an explicit validity mask.
    ///
    /// # Arguments
    /// * `embeddings` - The embedding tensor. Shape: `[batch, seq_len, embed_dim]`
    /// * `mask` - Validity mask. Shape: `[batch, seq_len]`. 1.0 = valid, 0.0 = padding.
    pub fn with_mask(embeddings: Tensor<B, 3>, mask: Tensor<B, 2>) -> Self {
        Self {
            embeddings,
            mask: Some(mask),
        }
    }

    /// Create a random representation for testing.
    ///
    /// Generates embeddings drawn from a standard normal distribution N(0, 1).
    ///
    /// # Arguments
    /// * `shape` - `[batch, seq_len, embed_dim]`
    /// * `device` - The device to create the tensor on
    pub fn random(shape: [usize; 3], device: &B::Device) -> Self {
        Self {
            embeddings: Tensor::random(shape, burn::tensor::Distribution::Normal(0.0, 1.0), device),
            mask: None,
        }
    }

    /// Create a zero-filled representation for testing.
    ///
    /// # Arguments
    /// * `shape` - `[batch, seq_len, embed_dim]`
    /// * `device` - The device to create the tensor on
    pub fn zeros(shape: [usize; 3], device: &B::Device) -> Self {
        Self {
            embeddings: Tensor::zeros(shape, device),
            mask: None,
        }
    }

    /// Returns `true` if this representation has a validity mask.
    pub fn has_mask(&self) -> bool {
        self.mask.is_some()
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.embeddings.dims()[0]
    }

    /// Get the sequence length (number of tokens/patches).
    pub fn seq_len(&self) -> usize {
        self.embeddings.dims()[1]
    }

    /// Get the embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embeddings.dims()[2]
    }

    /// Gather tokens at the given sequence indices from each batch element.
    ///
    /// Extracts a subset of tokens by index, producing a new representation
    /// with `seq_len = indices.len()`. This is used to extract context or
    /// target tokens according to a [`MaskSpec`].
    ///
    /// # Arguments
    /// * `indices` - Token indices to gather. Must all be < `self.seq_len()`.
    ///
    /// # Returns
    /// A new representation with shape `[batch, indices.len(), embed_dim]`.
    ///
    /// If the source representation carries a validity mask, the gathered
    /// representation preserves the corresponding mask entries.
    pub fn gather(&self, indices: &[usize]) -> Self {
        let [batch, _seq_len, embed_dim] = self.embeddings.dims();
        let num_indices = indices.len();
        let device = self.embeddings.device();

        if num_indices == 0 {
            let embeddings = Tensor::zeros([batch, 0, embed_dim], &device);
            let mask = self
                .mask
                .as_ref()
                .map(|_| Tensor::zeros([batch, 0], &device));
            return Self { embeddings, mask };
        }

        let index_data: Vec<i64> = indices.iter().map(|&index| index as i64).collect();
        let index_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(index_data, [num_indices]), &device);

        let embeddings = self.embeddings.clone().select(1, index_tensor.clone());
        let mask = self
            .mask
            .as_ref()
            .map(|mask| mask.clone().select(1, index_tensor));

        Self { embeddings, mask }
    }
}

/// Energy scalar measuring compatibility between two representations.
///
/// Lower energy indicates higher compatibility (better prediction).
/// This is the core quantity that JEPA training minimizes.
#[derive(Debug, Clone)]
pub struct Energy<B: Backend> {
    /// The energy value. Shape: scalar `[1]`
    pub value: Tensor<B, 1>,
}

/// Specification of which tokens are context (visible) vs target (hidden).
///
/// Generated by a [`MaskingStrategy`](crate::masking::MaskingStrategy) and consumed by the training loop
/// to split the input into context and prediction targets.
#[derive(Debug, Clone)]
pub struct MaskSpec {
    /// Indices of context (visible) tokens.
    pub context_indices: Vec<usize>,
    /// Indices of target (hidden) tokens to predict.
    pub target_indices: Vec<usize>,
    /// Total number of tokens in the input.
    pub total_tokens: usize,
}

impl MaskSpec {
    /// Verify that the mask is well-formed.
    pub fn validate(&self) -> Result<(), MaskError> {
        use std::collections::HashSet;

        // No empty partitions
        if self.context_indices.is_empty() {
            return Err(MaskError::EmptyContext);
        }
        if self.target_indices.is_empty() {
            return Err(MaskError::EmptyTarget);
        }

        // No duplicates
        let mut seen = HashSet::with_capacity(self.total_tokens);
        for &idx in self
            .context_indices
            .iter()
            .chain(self.target_indices.iter())
        {
            if idx >= self.total_tokens {
                return Err(MaskError::IndexOutOfBounds {
                    index: idx,
                    total: self.total_tokens,
                });
            }
            if !seen.insert(idx) {
                return Err(MaskError::DuplicateIndex(idx));
            }
        }

        Ok(())
    }

    /// Fraction of tokens masked as targets.
    ///
    /// Returns 0.0 if `total_tokens` is zero (avoids division by zero).
    pub fn mask_ratio(&self) -> f64 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        self.target_indices.len() as f64 / self.total_tokens as f64
    }
}

/// Errors related to mask validation.
#[derive(Debug, thiserror::Error)]
pub enum MaskError {
    #[error("context indices are empty, need at least one visible token")]
    EmptyContext,
    #[error("target indices are empty, need at least one token to predict")]
    EmptyTarget,
    #[error("index {index} is out of bounds for {total} total tokens")]
    IndexOutOfBounds { index: usize, total: usize },
    #[error("duplicate index {0} found in mask")]
    DuplicateIndex(usize),
}

/// Shape of the input to a masking strategy.
#[derive(Debug, Clone)]
pub enum InputShape {
    /// 2D grid of patches (for images).
    Image {
        /// Number of patch rows.
        height: usize,
        /// Number of patch columns.
        width: usize,
    },
    /// 3D grid of tubelets (for video).
    Video {
        /// Number of temporal positions.
        frames: usize,
        /// Number of patch rows per frame.
        height: usize,
        /// Number of patch columns per frame.
        width: usize,
    },
}

impl InputShape {
    /// Total number of tokens.
    pub fn total_tokens(&self) -> usize {
        match self {
            InputShape::Image { height, width } => height * width,
            InputShape::Video {
                frames,
                height,
                width,
            } => frames * height * width,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::ElementConversion;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

    #[test]
    fn test_representation_gather_selects_correct_tokens() {
        // Create representation with known values: token i has all values = i
        let data: Vec<f32> = (0..12)
            .map(|i| {
                let token = i / 4; // 3 tokens, embed_dim=4
                token as f32
            })
            .collect();
        let tensor = Tensor::<TestBackend, 3>::from_floats(
            burn::tensor::TensorData::new(data, [1, 3, 4]),
            &device(),
        );
        let repr = Representation::new(tensor);

        // Gather tokens 0 and 2 (skip token 1)
        let gathered = repr.gather(&[0, 2]);
        assert_eq!(gathered.batch_size(), 1);
        assert_eq!(gathered.seq_len(), 2);
        assert_eq!(gathered.embed_dim(), 4);

        let vals: Vec<f32> = gathered.embeddings.into_data().to_vec().unwrap();
        // Token 0 has all 0.0, token 2 has all 2.0
        assert!((vals[0] - 0.0).abs() < 1e-6);
        assert!((vals[4] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_representation_gather_preserves_batch() {
        let tensor = Tensor::<TestBackend, 3>::ones([3, 8, 16], &device());
        let repr = Representation::new(tensor);
        let gathered = repr.gather(&[1, 3, 5]);
        assert_eq!(gathered.batch_size(), 3);
        assert_eq!(gathered.seq_len(), 3);
        assert_eq!(gathered.embed_dim(), 16);
    }

    #[test]
    fn test_representation_gather_single_index() {
        let tensor = Tensor::<TestBackend, 3>::ones([2, 4, 8], &device());
        let repr = Representation::new(tensor);
        let gathered = repr.gather(&[2]);
        assert_eq!(gathered.seq_len(), 1);
    }

    #[test]
    fn test_representation_with_mask() {
        let embeddings = Tensor::<TestBackend, 3>::ones([2, 4, 8], &device());
        let mask = Tensor::<TestBackend, 2>::ones([2, 4], &device());
        let repr = Representation::with_mask(embeddings, mask);
        assert!(repr.has_mask());
        assert_eq!(repr.batch_size(), 2);
        assert_eq!(repr.seq_len(), 4);
        assert_eq!(repr.embed_dim(), 8);
    }

    #[test]
    fn test_representation_new_has_no_mask() {
        let repr = Representation::new(Tensor::<TestBackend, 3>::ones([1, 2, 4], &device()));
        assert!(!repr.has_mask());
    }

    #[test]
    fn test_representation_random() {
        let repr = Representation::<TestBackend>::random([2, 8, 16], &device());
        assert_eq!(repr.batch_size(), 2);
        assert_eq!(repr.seq_len(), 8);
        assert_eq!(repr.embed_dim(), 16);
        assert!(!repr.has_mask());
        // Should have non-zero values (random)
        let sum: f32 = repr.embeddings.abs().sum().into_scalar().elem();
        assert!(
            sum > 0.0,
            "random representation should have non-zero values"
        );
    }

    #[test]
    fn test_representation_zeros() {
        let repr = Representation::<TestBackend>::zeros([1, 4, 8], &device());
        assert_eq!(repr.batch_size(), 1);
        let sum: f32 = repr.embeddings.abs().sum().into_scalar().elem();
        assert!(sum < 1e-6, "zeros representation should be all zeros");
    }

    #[test]
    fn test_representation_gather_preserves_no_mask() {
        let repr = Representation::new(Tensor::<TestBackend, 3>::ones([1, 4, 8], &device()));
        let gathered = repr.gather(&[0, 2]);
        assert!(!gathered.has_mask());
    }

    #[test]
    fn test_representation_gather_preserves_mask() {
        let embeddings = Tensor::<TestBackend, 3>::ones([1, 4, 2], &device());
        let mask = Tensor::<TestBackend, 2>::from_floats([[1.0, 0.0, 1.0, 0.0]], &device());
        let repr = Representation::with_mask(embeddings, mask);

        let gathered = repr.gather(&[2, 1]);
        let gathered_mask = gathered
            .mask
            .expect("gathered representation should keep its mask");
        let values: Vec<f32> = gathered_mask.into_data().to_vec().unwrap();

        assert_eq!(values.len(), 2);
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mask_spec_validate_valid() {
        let mask = MaskSpec {
            context_indices: vec![0, 1, 2],
            target_indices: vec![3, 4, 5],
            total_tokens: 6,
        };
        assert!(mask.validate().is_ok());
    }

    #[test]
    fn test_mask_spec_validate_empty_context() {
        let mask = MaskSpec {
            context_indices: vec![],
            target_indices: vec![0, 1],
            total_tokens: 2,
        };
        assert!(matches!(mask.validate(), Err(MaskError::EmptyContext)));
    }

    #[test]
    fn test_mask_spec_validate_duplicate() {
        let mask = MaskSpec {
            context_indices: vec![0, 1],
            target_indices: vec![1, 2],
            total_tokens: 3,
        };
        assert!(matches!(mask.validate(), Err(MaskError::DuplicateIndex(1))));
    }

    #[test]
    fn test_mask_spec_validate_out_of_bounds() {
        let mask = MaskSpec {
            context_indices: vec![0],
            target_indices: vec![5],
            total_tokens: 3,
        };
        assert!(matches!(
            mask.validate(),
            Err(MaskError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_mask_ratio() {
        let mask = MaskSpec {
            context_indices: vec![0, 1, 2, 3],
            target_indices: vec![4, 5],
            total_tokens: 6,
        };
        assert!((mask.mask_ratio() - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mask_ratio_zero_total_tokens() {
        let mask = MaskSpec {
            context_indices: vec![],
            target_indices: vec![],
            total_tokens: 0,
        };
        // Should return 0.0 instead of NaN/panic
        assert_eq!(mask.mask_ratio(), 0.0);
    }

    #[test]
    fn test_representation_gather_empty_indices() {
        let tensor = Tensor::<TestBackend, 3>::ones([2, 4, 8], &device());
        let repr = Representation::new(tensor);
        let gathered = repr.gather(&[]);
        assert_eq!(gathered.batch_size(), 2);
        assert_eq!(gathered.seq_len(), 0);
        assert_eq!(gathered.embed_dim(), 8);
    }

    #[test]
    fn test_representation_gather_all_indices() {
        let tensor = Tensor::<TestBackend, 3>::ones([1, 3, 4], &device());
        let repr = Representation::new(tensor);
        let gathered = repr.gather(&[0, 1, 2]);
        assert_eq!(gathered.seq_len(), 3);
    }

    #[test]
    fn test_representation_gather_with_mask_empty_indices() {
        let embeddings = Tensor::<TestBackend, 3>::ones([1, 4, 2], &device());
        let mask = Tensor::<TestBackend, 2>::ones([1, 4], &device());
        let repr = Representation::with_mask(embeddings, mask);
        let gathered = repr.gather(&[]);
        assert_eq!(gathered.seq_len(), 0);
        assert!(gathered.has_mask());
    }

    #[test]
    fn test_mask_spec_validate_single_context_single_target() {
        let mask = MaskSpec {
            context_indices: vec![0],
            target_indices: vec![1],
            total_tokens: 2,
        };
        assert!(mask.validate().is_ok());
        assert!((mask.mask_ratio() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_input_shape_total_tokens() {
        assert_eq!(
            InputShape::Image {
                height: 14,
                width: 14
            }
            .total_tokens(),
            196
        );
        assert_eq!(
            InputShape::Video {
                frames: 8,
                height: 14,
                width: 14
            }
            .total_tokens(),
            1568
        );
    }
}
