use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};

use crate::cli::{ArchPreset, EnergyChoice, MaskingChoice, RegularizerChoice, TrainArgs};

pub const DEMO_IMAGE_COUNT: usize = 6;
const DEMO_IMAGE_SIZE: u32 = 96;

type PixelFn = fn(u32, u32) -> [u8; 3];

const DEMO_IMAGES: [(&str, PixelFn); DEMO_IMAGE_COUNT] = [
    ("class_a/gradient_h.png", gradient_horizontal),
    ("class_a/gradient_v.png", gradient_vertical),
    ("class_a/checker.png", checkerboard),
    ("class_b/diagonal.png", diagonal_mix),
    ("class_b/rings.png", concentric_rings),
    ("class_b/quadrants.png", quadrants),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoId {
    ImageFolderTraining,
    SyntheticTraining,
    PrepareImageFolder,
}

impl DemoId {
    pub const ALL: [DemoId; 3] = [
        DemoId::ImageFolderTraining,
        DemoId::SyntheticTraining,
        DemoId::PrepareImageFolder,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::ImageFolderTraining => "Image-Folder Training",
            Self::SyntheticTraining => "Synthetic Training",
            Self::PrepareImageFolder => "Prepare Demo Dataset",
        }
    }

    pub fn example_name(self) -> &'static str {
        match self {
            Self::ImageFolderTraining => "train_image_folder_demo",
            Self::SyntheticTraining => "train_synthetic_demo",
            Self::PrepareImageFolder => "prepare_demo_image_folder",
        }
    }

    pub fn subtitle(self) -> &'static str {
        match self {
            Self::ImageFolderTraining => {
                "Real strict I-JEPA optimization over generated image files"
            }
            Self::SyntheticTraining => {
                "Same optimizer and EMA path, but with synthetic random tensors"
            }
            Self::PrepareImageFolder => {
                "Generate a tiny recursive image dataset under target/example-data"
            }
        }
    }

    pub fn estimated_duration(self) -> &'static str {
        match self {
            Self::ImageFolderTraining => "~10-20s on CPU",
            Self::SyntheticTraining => "~10-20s on CPU",
            Self::PrepareImageFolder => "<1s",
        }
    }

    pub fn command(self) -> String {
        format!("cargo run -p jepa --example {}", self.example_name())
    }

    pub fn process_notes(self) -> &'static [&'static str] {
        match self {
            Self::ImageFolderTraining => &[
                "Generates a tiny nested image dataset at runtime.",
                "Exercises recursive file discovery and deterministic preprocessing.",
                "Runs the real strict masked-image training loop with AdamW and EMA.",
            ],
            Self::SyntheticTraining => &[
                "Skips dataset I/O and preprocessing.",
                "Still uses the real masking, optimizer, EMA, and predictor path.",
                "Good for a fast sanity check of the training stack.",
            ],
            Self::PrepareImageFolder => &[
                "Creates 6 small PNG fixtures under target/example-data/jepa/.",
                "Uses nested directories so --dataset-dir recursion is exercised.",
                "Keeps binary demo assets out of git while remaining reproducible.",
            ],
        }
    }

    pub fn monitoring_notes(self) -> &'static [&'static str] {
        match self {
            Self::ImageFolderTraining => &[
                "Watch loss and energy move step by step.",
                "Confirm the dataset path, file count, resize, and crop settings.",
                "Use the live log to follow preprocessing, training, and completion.",
            ],
            Self::SyntheticTraining => &[
                "Watch loss, learning rate, and EMA updates without disk I/O.",
                "Useful to confirm the strict reference path still executes cleanly.",
                "Compare behavior against the image-folder demo.",
            ],
            Self::PrepareImageFolder => &[
                "Watch the generated file list and output directory.",
                "Use the resulting path directly with jepa train --dataset-dir.",
                "This is a setup demo, not a learning run.",
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceDemoId {
    PatternVitSmall,
    PatternVitBase,
}

impl InferenceDemoId {
    pub const ALL: [InferenceDemoId; 2] = [
        InferenceDemoId::PatternVitSmall,
        InferenceDemoId::PatternVitBase,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::PatternVitSmall => "Pattern Walkthrough",
            Self::PatternVitBase => "Pattern Walkthrough XL",
        }
    }

    pub fn subtitle(self) -> &'static str {
        match self {
            Self::PatternVitSmall => "ViT-S/16 inference over deterministic demo image patterns",
            Self::PatternVitBase => {
                "ViT-B/16 inference over the same patterns to compare scale and cost"
            }
        }
    }

    pub fn estimated_duration(self) -> &'static str {
        match self {
            Self::PatternVitSmall => "~1-2s on CPU",
            Self::PatternVitBase => "~2-4s on CPU",
        }
    }

    pub fn preset(self) -> ArchPreset {
        match self {
            Self::PatternVitSmall => ArchPreset::VitSmall16,
            Self::PatternVitBase => ArchPreset::VitBase16,
        }
    }

    pub fn input_size(self) -> (usize, usize) {
        (224, 224)
    }

    pub fn sample_count(self) -> usize {
        3
    }

    pub fn process_notes(self) -> &'static [&'static str] {
        match self {
            Self::PatternVitSmall => &[
                "Builds a random-initialized ViT-S/16 encoder in demo mode.",
                "Synthesizes three deterministic image patterns used throughout the demos.",
                "Runs real tokenization and encoder attention, then inspects the embedding output.",
            ],
            Self::PatternVitBase => &[
                "Uses the same deterministic inputs with a larger ViT-B/16 encoder.",
                "Highlights how patch count stays fixed while latency and representation scale change.",
                "Good for comparing runtime cost against the smaller demo.",
            ],
        }
    }

    pub fn monitoring_notes(self) -> &'static [&'static str] {
        match self {
            Self::PatternVitSmall => &[
                "Watch the phase panel move from encoder init to per-sample inference.",
                "Latency, activation mean/std, and token norms update after each sample.",
                "The result panel explains what the embedding stats mean and what they do not.",
            ],
            Self::PatternVitBase => &[
                "Compare runtime and token-norm drift against the smaller pattern walkthrough.",
                "Use the sample previews to verify the same inputs flowed through both presets.",
                "This is a structure-and-monitoring demo, not a pretrained semantic benchmark.",
            ],
        }
    }

    pub fn engine_note(self) -> &'static str {
        match self {
            Self::PatternVitSmall | Self::PatternVitBase => {
                "Demo mode with random-initialized weights"
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PreparedDemoDataset {
    pub root: PathBuf,
    pub files: Vec<String>,
}

pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root should exist above crates/jepa")
        .to_path_buf()
}

pub fn demo_image_folder() -> PathBuf {
    workspace_root()
        .join("target")
        .join("example-data")
        .join("jepa")
        .join("demo-image-folder")
}

pub fn demo_checkpoint_dir(name: &str) -> PathBuf {
    workspace_root()
        .join("target")
        .join("example-data")
        .join("jepa")
        .join("checkpoints")
        .join(name)
}

pub fn prepare_demo_image_folder() -> Result<PreparedDemoDataset> {
    let root = demo_image_folder();
    if root.exists() {
        std::fs::remove_dir_all(&root)
            .with_context(|| format!("failed to clear {}", root.display()))?;
    }
    std::fs::create_dir_all(&root)
        .with_context(|| format!("failed to create {}", root.display()))?;

    let mut files = Vec::with_capacity(DEMO_IMAGE_COUNT);
    for (relative_path, pixel_fn) in DEMO_IMAGES {
        let path = root.join(relative_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }

        render_demo_image(pixel_fn)
            .save(&path)
            .with_context(|| format!("failed to save {}", path.display()))?;
        files.push(relative_path.to_string());
    }

    Ok(PreparedDemoDataset { root, files })
}

pub(crate) fn demo_pattern_images() -> Vec<(String, RgbImage)> {
    DEMO_IMAGES
        .iter()
        .map(|(relative_path, pixel_fn)| (relative_path.to_string(), render_demo_image(*pixel_fn)))
        .collect()
}

pub fn synthetic_demo_args() -> TrainArgs {
    TrainArgs {
        preset: ArchPreset::VitSmall16,
        steps: 2,
        warmup: 1,
        lr: 1e-3,
        batch_size: 2,
        dataset: None,
        dataset_key: "images".to_string(),
        dataset_dir: None,
        resize: None,
        crop_size: None,
        mean: None,
        std: None,
        dataset_limit: None,
        shuffle: false,
        masking: MaskingChoice::Block,
        energy: EnergyChoice::L2,
        regularizer: RegularizerChoice::Vicreg,
        reg_weight: 0.01,
        ema_momentum: 0.996,
        log_interval: 1,
        checkpoint_interval: 10,
        output_dir: demo_checkpoint_dir("synthetic-demo"),
    }
}

pub fn image_folder_demo_args(dataset_dir: PathBuf) -> TrainArgs {
    TrainArgs {
        preset: ArchPreset::VitSmall16,
        steps: 2,
        warmup: 1,
        lr: 1e-3,
        batch_size: 2,
        dataset: None,
        dataset_key: "images".to_string(),
        dataset_dir: Some(dataset_dir),
        resize: Some(256),
        crop_size: Some(224),
        mean: None,
        std: None,
        dataset_limit: Some(DEMO_IMAGE_COUNT),
        shuffle: true,
        masking: MaskingChoice::Block,
        energy: EnergyChoice::L2,
        regularizer: RegularizerChoice::Vicreg,
        reg_weight: 0.01,
        ema_momentum: 0.996,
        log_interval: 1,
        checkpoint_interval: 10,
        output_dir: demo_checkpoint_dir("image-folder-demo"),
    }
}

fn render_demo_image(pixel_fn: PixelFn) -> RgbImage {
    let mut image = RgbImage::new(DEMO_IMAGE_SIZE, DEMO_IMAGE_SIZE);
    for y in 0..DEMO_IMAGE_SIZE {
        for x in 0..DEMO_IMAGE_SIZE {
            image.put_pixel(x, y, Rgb(pixel_fn(x, y)));
        }
    }
    image
}

fn gradient_horizontal(x: u32, y: u32) -> [u8; 3] {
    let width = DEMO_IMAGE_SIZE - 1;
    let red = scale_u32_to_u8(x, width);
    let green = scale_u32_to_u8(y, width);
    let blue = 255u8.saturating_sub(red / 2);
    [red, green, blue]
}

fn gradient_vertical(x: u32, y: u32) -> [u8; 3] {
    let width = DEMO_IMAGE_SIZE - 1;
    let red = scale_u32_to_u8(y, width);
    let green = 255u8.saturating_sub(scale_u32_to_u8(x, width) / 2);
    let blue = scale_u32_to_u8(x + y, width * 2);
    [red, green, blue]
}

fn checkerboard(x: u32, y: u32) -> [u8; 3] {
    let tile = 12;
    let on = ((x / tile) + (y / tile)) % 2 == 0;
    if on {
        [240, 210, 40]
    } else {
        [20, 40, 180]
    }
}

fn diagonal_mix(x: u32, y: u32) -> [u8; 3] {
    let width = DEMO_IMAGE_SIZE - 1;
    let red = scale_u32_to_u8(x + y, width * 2);
    let green = scale_u32_to_u8(width.saturating_sub(x), width);
    let blue = scale_u32_to_u8(width.saturating_sub(y), width);
    [red, green, blue]
}

fn concentric_rings(x: u32, y: u32) -> [u8; 3] {
    let center = (DEMO_IMAGE_SIZE / 2) as i32;
    let dx = x as i32 - center;
    let dy = y as i32 - center;
    let distance_bucket = ((dx * dx + dy * dy) as f32).sqrt() as u32 / 6;
    match distance_bucket % 3 {
        0 => [230, 70, 70],
        1 => [40, 210, 120],
        _ => [60, 90, 230],
    }
}

fn quadrants(x: u32, y: u32) -> [u8; 3] {
    let half = DEMO_IMAGE_SIZE / 2;
    match (x < half, y < half) {
        (true, true) => [255, 110, 70],
        (false, true) => [80, 220, 120],
        (true, false) => [70, 140, 255],
        (false, false) => [240, 230, 120],
    }
}

fn scale_u32_to_u8(value: u32, max: u32) -> u8 {
    if max == 0 {
        return 0;
    }
    ((value.min(max) * 255) / max) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_id_all_has_three_variants() {
        assert_eq!(DemoId::ALL.len(), 3);
    }

    #[test]
    fn demo_id_titles_are_nonempty() {
        for demo in DemoId::ALL {
            assert!(!demo.title().is_empty());
            assert!(!demo.subtitle().is_empty());
            assert!(!demo.example_name().is_empty());
            assert!(!demo.estimated_duration().is_empty());
            assert!(!demo.command().is_empty());
            assert!(!demo.process_notes().is_empty());
            assert!(!demo.monitoring_notes().is_empty());
        }
    }

    #[test]
    fn demo_id_commands_contain_example_name() {
        for demo in DemoId::ALL {
            assert!(demo.command().contains(demo.example_name()));
        }
    }

    #[test]
    fn inference_demo_id_all_has_two_variants() {
        assert_eq!(InferenceDemoId::ALL.len(), 2);
    }

    #[test]
    fn inference_demo_id_properties() {
        for demo in InferenceDemoId::ALL {
            assert!(!demo.title().is_empty());
            assert!(!demo.subtitle().is_empty());
            assert!(!demo.estimated_duration().is_empty());
            assert!(!demo.engine_note().is_empty());
            assert!(!demo.process_notes().is_empty());
            assert!(!demo.monitoring_notes().is_empty());
            let (w, h) = demo.input_size();
            assert!(w > 0 && h > 0);
            assert!(demo.sample_count() > 0);
        }
    }

    #[test]
    fn scale_u32_to_u8_boundary_values() {
        assert_eq!(scale_u32_to_u8(0, 100), 0);
        assert_eq!(scale_u32_to_u8(100, 100), 255);
        assert_eq!(scale_u32_to_u8(50, 100), 127);
        assert_eq!(scale_u32_to_u8(0, 0), 0);
    }

    #[test]
    fn scale_u32_to_u8_clamps_above_max() {
        assert_eq!(scale_u32_to_u8(200, 100), 255);
    }

    #[test]
    fn render_demo_image_has_correct_dimensions() {
        let img = render_demo_image(gradient_horizontal);
        assert_eq!(img.width(), DEMO_IMAGE_SIZE);
        assert_eq!(img.height(), DEMO_IMAGE_SIZE);
    }

    #[test]
    fn all_pixel_functions_produce_valid_rgb() {
        let fns: &[PixelFn] = &[
            gradient_horizontal,
            gradient_vertical,
            checkerboard,
            diagonal_mix,
            concentric_rings,
            quadrants,
        ];
        for pixel_fn in fns {
            // Just ensure no panics for all coordinates
            for y in 0..DEMO_IMAGE_SIZE {
                for x in 0..DEMO_IMAGE_SIZE {
                    let _rgb = pixel_fn(x, y);
                }
            }
        }
    }

    #[test]
    fn demo_pattern_images_returns_all() {
        let images = demo_pattern_images();
        assert_eq!(images.len(), DEMO_IMAGE_COUNT);
        for (name, img) in &images {
            assert!(!name.is_empty());
            assert_eq!(img.width(), DEMO_IMAGE_SIZE);
            assert_eq!(img.height(), DEMO_IMAGE_SIZE);
        }
    }

    #[test]
    fn synthetic_demo_args_valid() {
        let args = synthetic_demo_args();
        assert!(args.warmup < args.steps);
        assert!(args.lr > 0.0);
        assert!(args.batch_size > 0);
    }

    #[test]
    fn workspace_root_exists() {
        let root = workspace_root();
        assert!(root.exists());
    }

    #[test]
    fn demo_image_folder_path_is_under_target() {
        let path = demo_image_folder();
        let path_str = path.display().to_string();
        assert!(path_str.contains("target"));
        assert!(path_str.contains("example-data"));
    }

    #[test]
    fn demo_checkpoint_dir_includes_name() {
        let path = demo_checkpoint_dir("my-demo");
        let path_str = path.display().to_string();
        assert!(path_str.contains("my-demo"));
    }

    #[test]
    fn checkerboard_pattern_is_deterministic() {
        let a = checkerboard(0, 0);
        let b = checkerboard(0, 0);
        assert_eq!(a, b);
        // On-tile at (0,0) should differ from off-tile at (12,0)
        let c = checkerboard(12, 0);
        assert_ne!(a, c);
    }

    #[test]
    fn quadrants_pattern_has_four_colors() {
        let tl = quadrants(0, 0);
        let tr = quadrants(DEMO_IMAGE_SIZE - 1, 0);
        let bl = quadrants(0, DEMO_IMAGE_SIZE - 1);
        let br = quadrants(DEMO_IMAGE_SIZE - 1, DEMO_IMAGE_SIZE - 1);
        // All four quadrants should be different
        assert_ne!(tl, tr);
        assert_ne!(tl, bl);
        assert_ne!(tl, br);
    }
}
