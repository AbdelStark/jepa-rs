//! Generate a tiny recursive image-folder dataset for `jepa train --dataset-dir`.
//!
//! The images are created under `target/example-data/jepa/demo-image-folder`
//! so the repository does not need checked-in binary assets for a smoke-demo.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p jepa --example prepare_demo_image_folder
//! ```

mod support;

use anyhow::Result;

fn main() -> Result<()> {
    let dataset_dir = support::ensure_demo_image_folder()?;

    println!("Prepared demo image folder at:");
    println!("  {}", dataset_dir.display());
    println!();
    println!("Images:");
    for relative_path in support::demo_image_relative_paths() {
        println!("  {}", relative_path);
    }
    println!();
    println!("Direct CLI run:");
    println!("  cargo run -p jepa -- train --preset vit-small-16 --steps 2 --batch-size 2 \\");
    println!(
        "    --dataset-dir {} --resize 256 --crop-size 224 --shuffle --dataset-limit {}",
        dataset_dir.display(),
        support::DEMO_IMAGE_COUNT
    );

    Ok(())
}
