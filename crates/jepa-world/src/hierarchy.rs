//! Hierarchical JEPA (H-JEPA) for multi-scale prediction.
//!
//! Implements RFC-010 (Hierarchical JEPA).
//!
//! H-JEPA stacks multiple JEPA levels, each operating at a different
//! temporal and spatial abstraction scale. Higher levels predict over
//! longer time horizons with coarser spatial resolution.

use burn::tensor::backend::Backend;

use jepa_core::types::Representation;
use jepa_core::{Encoder, Predictor};

/// A single level in the H-JEPA hierarchy.
///
/// Each level has its own encoder and predictor, and operates at
/// a specific temporal stride and spatial resolution.
pub struct JepaLevel<B: Backend> {
    /// Encoder for this level.
    pub encoder: Box<dyn Encoder<B, Input = Representation<B>>>,
    /// Predictor for this level.
    pub predictor: Box<dyn Predictor<B>>,
    /// Temporal abstraction factor.
    /// How many lower-level steps correspond to one step at this level.
    pub temporal_stride: usize,
}

/// Hierarchical JEPA with multiple abstraction levels.
///
/// Processes representations through a stack of JEPA levels,
/// where each level operates at progressively coarser temporal
/// and spatial scales.
///
/// # Example
///
/// ```
/// use jepa_world::hierarchy::HierarchicalJepa;
/// use burn_ndarray::NdArray;
///
/// type B = NdArray<f32>;
///
/// // An empty hierarchy is valid (zero levels)
/// let hjepa = HierarchicalJepa::<B>::new(vec![]);
/// assert_eq!(hjepa.num_levels(), 0);
/// ```
pub struct HierarchicalJepa<B: Backend> {
    /// Stack of JEPA levels, from finest (index 0) to coarsest.
    pub levels: Vec<JepaLevel<B>>,
}

impl<B: Backend> HierarchicalJepa<B> {
    /// Create a new hierarchical JEPA with the given levels.
    pub fn new(levels: Vec<JepaLevel<B>>) -> Self {
        Self { levels }
    }

    /// Get the number of hierarchy levels.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Encode input through all levels of the hierarchy.
    ///
    /// Returns representations at each level, from finest to coarsest.
    ///
    /// # Arguments
    /// * `input` - Input representation (typically from a base encoder)
    ///
    /// # Returns
    /// Vector of representations, one per level
    pub fn encode_all_levels(&self, input: &Representation<B>) -> Vec<Representation<B>> {
        let mut representations = Vec::with_capacity(self.levels.len());
        let mut current = input.clone();

        for level in &self.levels {
            let repr = level.encoder.encode(&current);
            representations.push(repr.clone());
            current = repr;
        }

        representations
    }

    /// Get the temporal stride at a specific level.
    ///
    /// The effective stride is the product of all strides up to
    /// and including the given level.
    ///
    /// # Panics
    ///
    /// Panics if `level_idx >= self.num_levels()`.
    pub fn effective_stride(&self, level_idx: usize) -> usize {
        assert!(
            level_idx < self.levels.len(),
            "level index {level_idx} out of bounds for hierarchy with {} levels",
            self.levels.len(),
        );
        self.levels[..=level_idx]
            .iter()
            .map(|l| l.temporal_stride)
            .product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn device() -> burn_ndarray::NdArrayDevice {
        burn_ndarray::NdArrayDevice::Cpu
    }

    /// Identity encoder for testing hierarchy.
    struct IdentityHierarchyEncoder {
        dim: usize,
    }

    impl Encoder<TestBackend> for IdentityHierarchyEncoder {
        type Input = Representation<TestBackend>;

        fn encode(&self, input: &Self::Input) -> Representation<TestBackend> {
            input.clone()
        }

        fn embed_dim(&self) -> usize {
            self.dim
        }
    }

    /// Zero predictor for testing hierarchy.
    struct ZeroPredictorH {
        embed_dim: usize,
    }

    impl Predictor<TestBackend> for ZeroPredictorH {
        fn predict(
            &self,
            _context: &Representation<TestBackend>,
            target_positions: &Tensor<TestBackend, 2>,
            _latent: Option<&Tensor<TestBackend, 2>>,
        ) -> Representation<TestBackend> {
            let [batch, num_targets] = target_positions.dims();
            let device = target_positions.device();
            Representation::new(Tensor::zeros([batch, num_targets, self.embed_dim], &device))
        }
    }

    fn make_level(dim: usize, stride: usize) -> JepaLevel<TestBackend> {
        JepaLevel {
            encoder: Box::new(IdentityHierarchyEncoder { dim }),
            predictor: Box::new(ZeroPredictorH { embed_dim: dim }),
            temporal_stride: stride,
        }
    }

    #[test]
    fn test_hierarchy_num_levels() {
        let hjepa = HierarchicalJepa::new(vec![
            make_level(64, 1),
            make_level(64, 2),
            make_level(64, 4),
        ]);
        assert_eq!(hjepa.num_levels(), 3);
    }

    #[test]
    fn test_hierarchy_encode_all_levels() {
        let hjepa = HierarchicalJepa::new(vec![make_level(32, 1), make_level(32, 2)]);

        let input = Representation::new(Tensor::ones([1, 8, 32], &device()));
        let reprs = hjepa.encode_all_levels(&input);

        assert_eq!(reprs.len(), 2);
        assert_eq!(reprs[0].seq_len(), 8);
        assert_eq!(reprs[1].seq_len(), 8); // identity encoder preserves shape
    }

    #[test]
    #[should_panic(expected = "level index 3 out of bounds")]
    fn test_effective_stride_out_of_bounds() {
        let hjepa = HierarchicalJepa::new(vec![
            make_level(64, 2),
            make_level(64, 3),
            make_level(64, 4),
        ]);
        let _ = hjepa.effective_stride(3);
    }

    #[test]
    fn test_effective_stride() {
        let hjepa = HierarchicalJepa::new(vec![
            make_level(64, 2),
            make_level(64, 3),
            make_level(64, 4),
        ]);

        assert_eq!(hjepa.effective_stride(0), 2);
        assert_eq!(hjepa.effective_stride(1), 6); // 2 * 3
        assert_eq!(hjepa.effective_stride(2), 24); // 2 * 3 * 4
    }
}
