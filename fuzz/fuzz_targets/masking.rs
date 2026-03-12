#![no_main]

use jepa_core::masking::{BlockMasking, MaskingStrategy, MultiBlockMasking, SpatiotemporalMasking};
use jepa_core::types::InputShape;
use libfuzzer_sys::fuzz_target;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let mut seed = [0u8; 32];
    for (index, byte) in seed.iter_mut().enumerate() {
        *byte = data.get(index).copied().unwrap_or(index as u8);
    }
    let mut rng = ChaCha8Rng::from_seed(seed);

    match data[0] % 3 {
        0 => {
            let shape = InputShape::Image {
                height: 2 + usize::from(data.get(1).copied().unwrap_or(0) % 31),
                width: 2 + usize::from(data.get(2).copied().unwrap_or(0) % 31),
            };
            let min_scale = 0.05 + f64::from(data.get(3).copied().unwrap_or(0) % 20) / 100.0;
            let max_scale = min_scale
                + f64::from(data.get(4).copied().unwrap_or(0) % 30) / 100.0;
            let masking = BlockMasking {
                num_targets: 1 + usize::from(data.get(5).copied().unwrap_or(0) % 8),
                target_scale: (min_scale, max_scale.min(0.95)),
                target_aspect_ratio: (0.5, 2.0),
            };
            let mask = masking.generate_mask(&shape, &mut rng);
            let _ = mask.validate();
        }
        1 => {
            let shape = InputShape::Video {
                frames: 1 + usize::from(data.get(1).copied().unwrap_or(0) % 6),
                height: 2 + usize::from(data.get(2).copied().unwrap_or(0) % 16),
                width: 2 + usize::from(data.get(3).copied().unwrap_or(0) % 16),
            };
            let masking = SpatiotemporalMasking {
                num_targets: 1 + usize::from(data.get(4).copied().unwrap_or(0) % 6),
                temporal_extent: (1, 1 + usize::from(data.get(5).copied().unwrap_or(0) % 4)),
                spatial_scale: (0.05, 0.35),
            };
            let mask = masking.generate_mask(&shape, &mut rng);
            let _ = mask.validate();
        }
        _ => {
            let shape = InputShape::Video {
                frames: 1 + usize::from(data.get(1).copied().unwrap_or(0) % 4),
                height: 2 + usize::from(data.get(2).copied().unwrap_or(0) % 12),
                width: 2 + usize::from(data.get(3).copied().unwrap_or(0) % 12),
            };
            let ratio = 0.05 + f64::from(data.get(4).copied().unwrap_or(0) % 80) / 100.0;
            let masking = MultiBlockMasking {
                mask_ratio: ratio.min(0.95),
                num_blocks: 1 + usize::from(data.get(5).copied().unwrap_or(0) % 8),
            };
            let mask = masking.generate_mask(&shape, &mut rng);
            let _ = mask.validate();
        }
    }
});
