//! Masking strategies for JEPA.
//!
//! Implements RFC-005 (Masking Strategies).
//!
//! Masking determines which parts of the input are context (visible)
//! and which are targets (to predict). This is the most critical
//! design decision in JEPA, as the masking strategy determines what
//! the model learns.

use rand::{Rng, RngExt as _};

use crate::types::{InputShape, MaskSpec};

/// Trait for masking strategies.
///
/// A masking strategy generates a [`MaskSpec`] that partitions input tokens
/// into context (visible) and target (hidden) sets.
///
/// # Example
///
/// ```
/// use jepa_core::masking::{MaskingStrategy, BlockMasking};
/// use jepa_core::types::InputShape;
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
///
/// let masking = BlockMasking {
///     num_targets: 4,
///     target_scale: (0.15, 0.2),
///     target_aspect_ratio: (0.75, 1.5),
/// };
/// let shape = InputShape::Image { height: 14, width: 14 };
/// let mut rng = ChaCha8Rng::seed_from_u64(42);
/// let mask = masking.generate_mask(&shape, &mut rng);
///
/// assert!(mask.validate().is_ok());
/// assert_eq!(mask.context_indices.len() + mask.target_indices.len(), 196);
/// ```
pub trait MaskingStrategy {
    /// Generate a mask for a given input shape.
    ///
    /// # Arguments
    /// * `shape` - The shape of the input (image grid or video grid)
    /// * `rng` - Random number generator for stochastic masking
    ///
    /// # Returns
    /// A [`MaskSpec`] with disjoint context and target index sets
    fn generate_mask(&self, shape: &InputShape, rng: &mut impl Rng) -> MaskSpec;
}

/// Block masking for images (I-JEPA style).
///
/// Masks one or more contiguous rectangular blocks as targets,
/// with the remaining patches as context. This forces the model
/// to predict large semantic regions from partial observations.
pub struct BlockMasking {
    /// Number of target blocks to mask.
    pub num_targets: usize,
    /// Target block scale range as fraction of total patches: `(min, max)`.
    pub target_scale: (f64, f64),
    /// Target block aspect ratio range: `(min, max)`.
    pub target_aspect_ratio: (f64, f64),
}

impl MaskingStrategy for BlockMasking {
    fn generate_mask(&self, shape: &InputShape, rng: &mut impl Rng) -> MaskSpec {
        let (height, width) = match shape {
            InputShape::Image { height, width } => (*height, *width),
            InputShape::Video {
                height,
                width,
                frames: _,
            } => (*height, *width),
        };
        let total = height * width;

        let mut target_set = std::collections::HashSet::new();

        for _ in 0..self.num_targets {
            // Sample scale and aspect ratio
            let scale = self.target_scale.0
                + rng.random::<f64>() * (self.target_scale.1 - self.target_scale.0);
            let aspect = self.target_aspect_ratio.0
                + rng.random::<f64>() * (self.target_aspect_ratio.1 - self.target_aspect_ratio.0);

            // Compute block dimensions
            let num_patches = (total as f64 * scale / self.num_targets as f64).round() as usize;
            let block_h = ((num_patches as f64 * aspect).sqrt()).round() as usize;
            let block_w = if block_h > 0 {
                (num_patches / block_h).max(1)
            } else {
                1
            };

            let block_h = block_h.clamp(1, height);
            let block_w = block_w.clamp(1, width);

            // Random top-left corner
            let top = rng.random_range(0..=(height - block_h));
            let left = rng.random_range(0..=(width - block_w));

            for r in top..(top + block_h) {
                for c in left..(left + block_w) {
                    target_set.insert(r * width + c);
                }
            }
        }

        // Ensure we have at least one context token
        if target_set.len() >= total {
            if let Some(&first) = target_set.iter().next() {
                target_set.remove(&first);
            }
        }

        let mut target_indices: Vec<usize> = target_set.into_iter().collect();
        target_indices.sort_unstable();

        let target_set_lookup: std::collections::HashSet<usize> =
            target_indices.iter().copied().collect();
        let context_indices: Vec<usize> = (0..total)
            .filter(|i| !target_set_lookup.contains(i))
            .collect();

        MaskSpec {
            context_indices,
            target_indices,
            total_tokens: total,
        }
    }
}

/// Spatiotemporal masking for video (V-JEPA style).
///
/// Masks contiguous 3D regions in space and time, forcing the model
/// to predict temporal dynamics and spatial structure jointly.
pub struct SpatiotemporalMasking {
    /// Number of target tubes to mask.
    pub num_targets: usize,
    /// Temporal extent range of each tube in frames: `(min, max)`.
    pub temporal_extent: (usize, usize),
    /// Spatial scale of each tube as fraction of frame area: `(min, max)`.
    pub spatial_scale: (f64, f64),
}

impl MaskingStrategy for SpatiotemporalMasking {
    fn generate_mask(&self, shape: &InputShape, rng: &mut impl Rng) -> MaskSpec {
        let (frames, height, width) = match shape {
            InputShape::Video {
                frames,
                height,
                width,
            } => (*frames, *height, *width),
            InputShape::Image { height, width } => (1, *height, *width),
        };
        let total = frames * height * width;
        let frame_area = height * width;

        let mut target_set = std::collections::HashSet::new();

        for _ in 0..self.num_targets {
            // Sample temporal extent
            let t_extent = rng.random_range(self.temporal_extent.0..=self.temporal_extent.1);
            let t_extent = t_extent.clamp(1, frames);

            // Sample spatial block
            let scale = self.spatial_scale.0
                + rng.random::<f64>() * (self.spatial_scale.1 - self.spatial_scale.0);
            let num_spatial = (frame_area as f64 * scale).round() as usize;
            let block_side = (num_spatial as f64).sqrt().round() as usize;
            let block_h = block_side.clamp(1, height);
            let block_w = block_side.clamp(1, width);

            let t_start = rng.random_range(0..=(frames - t_extent));
            let top = rng.random_range(0..=(height - block_h));
            let left = rng.random_range(0..=(width - block_w));

            for t in t_start..(t_start + t_extent) {
                for r in top..(top + block_h) {
                    for c in left..(left + block_w) {
                        target_set.insert(t * frame_area + r * width + c);
                    }
                }
            }
        }

        // Ensure non-empty context
        if target_set.len() >= total {
            if let Some(&first) = target_set.iter().next() {
                target_set.remove(&first);
            }
        }

        let mut target_indices: Vec<usize> = target_set.into_iter().collect();
        target_indices.sort_unstable();

        let target_set_lookup: std::collections::HashSet<usize> =
            target_indices.iter().copied().collect();
        let context_indices: Vec<usize> = (0..total)
            .filter(|i| !target_set_lookup.contains(i))
            .collect();

        MaskSpec {
            context_indices,
            target_indices,
            total_tokens: total,
        }
    }
}

/// Multi-block masking (V-JEPA 2 style).
///
/// Masks multiple blocks with specific constraints on total coverage ratio.
pub struct MultiBlockMasking {
    /// Target masking ratio (fraction of tokens masked).
    pub mask_ratio: f64,
    /// Number of mask blocks.
    pub num_blocks: usize,
}

impl MaskingStrategy for MultiBlockMasking {
    fn generate_mask(&self, shape: &InputShape, rng: &mut impl Rng) -> MaskSpec {
        let (height, width) = match shape {
            InputShape::Image { height, width } => (*height, *width),
            InputShape::Video {
                height,
                width,
                frames: _,
            } => (*height, *width),
        };
        let total = shape.total_tokens();
        let target_count = ((total as f64) * self.mask_ratio).round() as usize;
        let per_block = (target_count / self.num_blocks).max(1);

        let mut target_set = std::collections::HashSet::new();

        for _ in 0..self.num_blocks {
            let block_side = (per_block as f64).sqrt().round() as usize;
            let block_h = block_side.clamp(1, height);
            let block_w = block_side.clamp(1, width);

            let top = rng.random_range(0..=(height - block_h));
            let left = rng.random_range(0..=(width - block_w));

            for r in top..(top + block_h) {
                for c in left..(left + block_w) {
                    target_set.insert(r * width + c);
                }
            }
        }

        // Ensure non-empty context
        if target_set.len() >= total {
            if let Some(&first) = target_set.iter().next() {
                target_set.remove(&first);
            }
        }
        // Ensure non-empty target
        if target_set.is_empty() {
            target_set.insert(rng.random_range(0..total));
        }

        let mut target_indices: Vec<usize> = target_set.into_iter().collect();
        target_indices.sort_unstable();

        let target_set_lookup: std::collections::HashSet<usize> =
            target_indices.iter().copied().collect();
        let context_indices: Vec<usize> = (0..total)
            .filter(|i| !target_set_lookup.contains(i))
            .collect();

        MaskSpec {
            context_indices,
            target_indices,
            total_tokens: total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }

    #[test]
    fn test_block_masking_partitions_all_patches() {
        let masking = BlockMasking {
            num_targets: 4,
            target_scale: (0.15, 0.2),
            target_aspect_ratio: (0.75, 1.5),
        };
        let shape = InputShape::Image {
            height: 14,
            width: 14,
        };
        let mask = masking.generate_mask(&shape, &mut rng(42));

        // Context + target should cover all tokens (no overlap ensured by construction)
        assert!(mask.validate().is_ok());
        assert_eq!(mask.context_indices.len() + mask.target_indices.len(), 196);
    }

    #[test]
    fn test_block_masking_non_empty_partitions() {
        let masking = BlockMasking {
            num_targets: 4,
            target_scale: (0.15, 0.2),
            target_aspect_ratio: (0.75, 1.5),
        };
        let shape = InputShape::Image {
            height: 14,
            width: 14,
        };
        let mask = masking.generate_mask(&shape, &mut rng(42));
        assert!(!mask.context_indices.is_empty());
        assert!(!mask.target_indices.is_empty());
    }

    #[test]
    fn test_block_masking_no_overlap() {
        let masking = BlockMasking {
            num_targets: 4,
            target_scale: (0.15, 0.2),
            target_aspect_ratio: (0.75, 1.5),
        };
        let shape = InputShape::Image {
            height: 14,
            width: 14,
        };
        let mask = masking.generate_mask(&shape, &mut rng(42));
        let context_set: std::collections::HashSet<_> = mask.context_indices.iter().collect();
        for t in &mask.target_indices {
            assert!(!context_set.contains(t), "overlap at index {t}");
        }
    }

    #[test]
    fn test_masking_reproducible_with_same_seed() {
        let masking = BlockMasking {
            num_targets: 4,
            target_scale: (0.15, 0.2),
            target_aspect_ratio: (0.75, 1.5),
        };
        let shape = InputShape::Image {
            height: 14,
            width: 14,
        };
        let mask1 = masking.generate_mask(&shape, &mut rng(42));
        let mask2 = masking.generate_mask(&shape, &mut rng(42));
        assert_eq!(mask1.context_indices, mask2.context_indices);
        assert_eq!(mask1.target_indices, mask2.target_indices);
    }

    #[test]
    fn test_masking_different_with_different_seeds() {
        let masking = BlockMasking {
            num_targets: 4,
            target_scale: (0.15, 0.2),
            target_aspect_ratio: (0.75, 1.5),
        };
        let shape = InputShape::Image {
            height: 14,
            width: 14,
        };
        let mask1 = masking.generate_mask(&shape, &mut rng(42));
        let mask2 = masking.generate_mask(&shape, &mut rng(43));
        assert_ne!(mask1.target_indices, mask2.target_indices);
    }

    #[test]
    fn test_spatiotemporal_masking_valid() {
        let masking = SpatiotemporalMasking {
            num_targets: 2,
            temporal_extent: (2, 4),
            spatial_scale: (0.1, 0.2),
        };
        let shape = InputShape::Video {
            frames: 8,
            height: 14,
            width: 14,
        };
        let mask = masking.generate_mask(&shape, &mut rng(42));
        assert!(mask.validate().is_ok());
        assert!(!mask.context_indices.is_empty());
        assert!(!mask.target_indices.is_empty());
    }

    #[test]
    fn test_multi_block_masking_valid() {
        let masking = MultiBlockMasking {
            mask_ratio: 0.5,
            num_blocks: 4,
        };
        let shape = InputShape::Image {
            height: 14,
            width: 14,
        };
        let mask = masking.generate_mask(&shape, &mut rng(42));
        assert!(mask.validate().is_ok());
        assert!(!mask.context_indices.is_empty());
        assert!(!mask.target_indices.is_empty());
    }

    // --- Edge-case tests ---

    #[test]
    fn test_block_masking_minimum_grid_2x2() {
        // Smallest grid where block masking can produce both context and target
        let masking = BlockMasking {
            num_targets: 1,
            target_scale: (0.25, 0.5),
            target_aspect_ratio: (1.0, 1.0),
        };
        let shape = InputShape::Image {
            height: 2,
            width: 2,
        };
        let mask = masking.generate_mask(&shape, &mut rng(42));
        assert!(mask.validate().is_ok());
        assert!(!mask.context_indices.is_empty());
        assert!(!mask.target_indices.is_empty());
        assert_eq!(mask.context_indices.len() + mask.target_indices.len(), 4);
    }

    #[test]
    fn test_block_masking_maximum_coverage() {
        // Many targets with high scale — tests the non-empty context guarantee
        let masking = BlockMasking {
            num_targets: 10,
            target_scale: (0.8, 0.99),
            target_aspect_ratio: (0.5, 2.0),
        };
        let shape = InputShape::Image {
            height: 4,
            width: 4,
        };
        let mask = masking.generate_mask(&shape, &mut rng(42));
        assert!(mask.validate().is_ok());
        assert!(
            !mask.context_indices.is_empty(),
            "must always have at least one context token"
        );
    }

    #[test]
    fn test_multi_block_masking_very_high_ratio() {
        // mask_ratio near 1.0 — tests the non-empty context guarantee
        let masking = MultiBlockMasking {
            mask_ratio: 0.99,
            num_blocks: 8,
        };
        let shape = InputShape::Image {
            height: 4,
            width: 4,
        };
        let mask = masking.generate_mask(&shape, &mut rng(42));
        assert!(mask.validate().is_ok());
        assert!(!mask.context_indices.is_empty());
        assert!(!mask.target_indices.is_empty());
    }

    #[test]
    fn test_spatiotemporal_masking_single_frame() {
        // Video with 1 frame — degenerates to image-like behavior
        let masking = SpatiotemporalMasking {
            num_targets: 1,
            temporal_extent: (1, 1),
            spatial_scale: (0.1, 0.2),
        };
        let shape = InputShape::Video {
            frames: 1,
            height: 8,
            width: 8,
        };
        let mask = masking.generate_mask(&shape, &mut rng(42));
        assert!(mask.validate().is_ok());
        assert_eq!(mask.context_indices.len() + mask.target_indices.len(), 64);
    }

    #[test]
    fn test_spatiotemporal_masking_on_image_shape() {
        // Image shape passed to spatiotemporal masking (1 frame fallback)
        let masking = SpatiotemporalMasking {
            num_targets: 2,
            temporal_extent: (1, 1),
            spatial_scale: (0.1, 0.2),
        };
        let shape = InputShape::Image {
            height: 8,
            width: 8,
        };
        let mask = masking.generate_mask(&shape, &mut rng(42));
        assert!(mask.validate().is_ok());
        assert_eq!(mask.context_indices.len() + mask.target_indices.len(), 64);
    }

    // --- Property-based tests ---

    proptest! {
        #[test]
        fn prop_block_mask_always_valid(
            seed in 0u64..100000,
            grid_h in 4usize..20,
            grid_w in 4usize..20,
            num_targets in 1usize..6,
        ) {
            let masking = BlockMasking {
                num_targets,
                target_scale: (0.1, 0.3),
                target_aspect_ratio: (0.75, 1.5),
            };
            let shape = InputShape::Image { height: grid_h, width: grid_w };
            let mask = masking.generate_mask(&shape, &mut rng(seed));

            // Mask should always be valid
            prop_assert!(mask.validate().is_ok());

            // Context + target = total, no overlap
            let total = grid_h * grid_w;
            prop_assert_eq!(mask.context_indices.len() + mask.target_indices.len(), total);
            prop_assert!(!mask.context_indices.is_empty());
            prop_assert!(!mask.target_indices.is_empty());

            // No duplicates in context
            let mut ctx = mask.context_indices.clone();
            ctx.sort_unstable();
            ctx.dedup();
            prop_assert_eq!(ctx.len(), mask.context_indices.len());

            // No duplicates in target
            let mut tgt = mask.target_indices.clone();
            tgt.sort_unstable();
            tgt.dedup();
            prop_assert_eq!(tgt.len(), mask.target_indices.len());

            // All indices in bounds
            for &i in &mask.context_indices {
                prop_assert!(i < total);
            }
            for &i in &mask.target_indices {
                prop_assert!(i < total);
            }
        }

        #[test]
        fn prop_spatiotemporal_mask_always_valid(
            seed in 0u64..100000,
            frames in 4usize..12,
            grid_h in 4usize..12,
            grid_w in 4usize..12,
        ) {
            let masking = SpatiotemporalMasking {
                num_targets: 2,
                temporal_extent: (2, 3),
                spatial_scale: (0.05, 0.15),
            };
            let shape = InputShape::Video { frames, height: grid_h, width: grid_w };
            let mask = masking.generate_mask(&shape, &mut rng(seed));

            prop_assert!(mask.validate().is_ok());

            let total = frames * grid_h * grid_w;
            prop_assert_eq!(mask.context_indices.len() + mask.target_indices.len(), total);
            prop_assert!(!mask.context_indices.is_empty());
            prop_assert!(!mask.target_indices.is_empty());
        }

        #[test]
        fn prop_multi_block_mask_always_valid(
            seed in 0u64..100000,
            grid_h in 4usize..16,
            grid_w in 4usize..16,
            mask_ratio in 0.1f64..0.8,
            num_blocks in 1usize..6,
        ) {
            let masking = MultiBlockMasking { mask_ratio, num_blocks };
            let shape = InputShape::Image { height: grid_h, width: grid_w };
            let mask = masking.generate_mask(&shape, &mut rng(seed));

            prop_assert!(mask.validate().is_ok());
            prop_assert!(!mask.context_indices.is_empty());
            prop_assert!(!mask.target_indices.is_empty());
        }

        #[test]
        fn prop_masking_is_deterministic(seed in 0u64..100000) {
            let masking = BlockMasking {
                num_targets: 4,
                target_scale: (0.15, 0.2),
                target_aspect_ratio: (0.75, 1.5),
            };
            let shape = InputShape::Image { height: 14, width: 14 };
            let mask1 = masking.generate_mask(&shape, &mut rng(seed));
            let mask2 = masking.generate_mask(&shape, &mut rng(seed));
            prop_assert_eq!(mask1.context_indices, mask2.context_indices);
            prop_assert_eq!(mask1.target_indices, mask2.target_indices);
        }
    }
}
