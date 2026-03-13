//! Run the real `jepa train` image-folder path against a tiny generated dataset.
//!
//! This example creates a recursive dataset directory with a handful of PNG
//! files under `target/example-data/jepa/demo-image-folder`, then feeds that
//! directory into the same strict training command used by the CLI.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p jepa --example train_image_folder_demo
//! ```

mod support;

use anyhow::Result;
use jepa::cli::{ArchPreset, EnergyChoice, MaskingChoice, RegularizerChoice, TrainArgs};

fn main() -> Result<()> {
    let dataset_dir = support::ensure_demo_image_folder()?;

    println!("Using demo image folder dataset:");
    println!("  {}", dataset_dir.display());
    println!("  {} images", support::DEMO_IMAGE_COUNT);
    println!();

    jepa::commands::train::run(TrainArgs {
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
        dataset_limit: Some(support::DEMO_IMAGE_COUNT),
        shuffle: true,
        masking: MaskingChoice::Block,
        energy: EnergyChoice::L2,
        regularizer: RegularizerChoice::Vicreg,
        reg_weight: 0.01,
        ema_momentum: 0.996,
        log_interval: 1,
        checkpoint_interval: 10,
        output_dir: support::demo_checkpoint_dir("image-folder-demo"),
    })
}
