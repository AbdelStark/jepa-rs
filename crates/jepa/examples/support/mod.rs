#![allow(dead_code)]

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::{Rgb, RgbImage};

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

pub fn ensure_demo_image_folder() -> Result<PathBuf> {
    let root = demo_image_folder();
    if root.exists() {
        std::fs::remove_dir_all(&root)
            .with_context(|| format!("failed to clear {}", root.display()))?;
    }
    std::fs::create_dir_all(&root)
        .with_context(|| format!("failed to create {}", root.display()))?;

    for (relative_path, pixel_fn) in DEMO_IMAGES {
        let path = root.join(relative_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }

        render_demo_image(pixel_fn)
            .save(&path)
            .with_context(|| format!("failed to save {}", path.display()))?;
    }

    Ok(root)
}

pub fn demo_image_relative_paths() -> impl Iterator<Item = &'static str> {
    DEMO_IMAGES.iter().map(|(path, _)| *path)
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
