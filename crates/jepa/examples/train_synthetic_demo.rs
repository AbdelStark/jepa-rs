//! Run the real `jepa train` synthetic fallback path.
//!
//! This keeps the full strict masked-image optimizer/EMA training loop, but
//! uses random tensors instead of loading a dataset from disk.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p jepa --example train_synthetic_demo
//! ```

mod support;

use anyhow::Result;
use jepa::cli::{ArchPreset, EnergyChoice, MaskingChoice, RegularizerChoice, TrainArgs};

fn main() -> Result<()> {
    jepa::commands::train::run(TrainArgs {
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
        output_dir: support::demo_checkpoint_dir("synthetic-demo"),
    })
}
