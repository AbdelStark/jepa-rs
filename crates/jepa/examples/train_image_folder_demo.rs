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

use anyhow::Result;

fn main() -> Result<()> {
    let prepared = jepa::demo::prepare_demo_image_folder()?;

    println!("Using demo image folder dataset:");
    println!("  {}", prepared.root.display());
    println!("  {} images", jepa::demo::DEMO_IMAGE_COUNT);
    println!();

    jepa::commands::train::run(jepa::demo::image_folder_demo_args(prepared.root))
}
