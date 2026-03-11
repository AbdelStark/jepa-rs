//! Masking strategies for JEPA.
//!
//! Implements RFC-005 (Masking Strategies).
//!
//! Masking determines which parts of the input are context (visible)
//! and which are targets (to predict). This is the most critical
//! design decision in JEPA, as the masking strategy determines what
//! the model learns.

use rand::Rng;

use crate::types::{InputShape, MaskSpec};

/// Trait for masking strategies.
///
/// A masking strategy generates a [`MaskSpec`] that partitions input tokens
/// into context (visible) and target (hidden) sets.
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
                + rng.gen::<f64>() * (self.target_scale.1 - self.target_scale.0);
            let aspect = self.target_aspect_ratio.0
                + rng.gen::<f64>() * (self.target_aspect_ratio.1 - self.target_aspect_ratio.0);

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
            let top = rng.gen_range(0..=(height - block_h));
            let left = rng.gen_range(0..=(width - block_w));

            for r in top..(top + block_h) {
                for c in left..(left + block_w) {
                    target_set.insert(r * width + c);
                }
            }
        }

        // Ensure we have at least one context token
        if target_set.len() >= total {
            // Remove one random target to ensure non-empty context
            let first = *target_set.iter().next().unwrap();
            target_set.remove(&first);
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
            let t_extent = rng.gen_range(self.temporal_extent.0..=self.temporal_extent.1);
            let t_extent = t_extent.clamp(1, frames);

            // Sample spatial block
            let scale = self.spatial_scale.0
                + rng.gen::<f64>() * (self.spatial_scale.1 - self.spatial_scale.0);
            let num_spatial = (frame_area as f64 * scale).round() as usize;
            let block_side = (num_spatial as f64).sqrt().round() as usize;
            let block_h = block_side.clamp(1, height);
            let block_w = block_side.clamp(1, width);

            let t_start = rng.gen_range(0..=(frames - t_extent));
            let top = rng.gen_range(0..=(height - block_h));
            let left = rng.gen_range(0..=(width - block_w));

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
            let first = *target_set.iter().next().unwrap();
            target_set.remove(&first);
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

            let top = rng.gen_range(0..=(height - block_h));
            let left = rng.gen_range(0..=(width - block_w));

            for r in top..(top + block_h) {
                for c in left..(left + block_w) {
                    target_set.insert(r * width + c);
                }
            }
        }

        // Ensure non-empty context
        if target_set.len() >= total {
            let first = *target_set.iter().next().unwrap();
            target_set.remove(&first);
        }
        // Ensure non-empty target
        if target_set.is_empty() {
            target_set.insert(rng.gen_range(0..total));
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
}
