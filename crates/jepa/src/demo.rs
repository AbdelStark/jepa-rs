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
