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

use anyhow::Result;

fn main() -> Result<()> {
    jepa::commands::train::run(jepa::demo::synthetic_demo_args())
}
