use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// jepa — the unified CLI for the jepa-rs workspace
///
/// Run without a subcommand to launch the interactive TUI dashboard.
#[derive(Parser)]
#[command(
    name = "jepa",
    version,
    about = "JEPA toolkit — train, inspect, and run Joint-Embedding Predictive Architecture models",
    long_about = None,
    after_help = "Run `jepa` with no subcommand to launch the interactive TUI."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand)]
pub enum Command {
    /// List available pretrained models in the registry
    Models(ModelsArgs),
    /// Inspect a model file (safetensors or ONNX metadata)
    Inspect(InspectArgs),
    /// Checkpoint operations (inspect metadata, convert)
    Checkpoint(CheckpointArgs),
    /// Launch a training run
    Train(TrainArgs),
    /// Encode inputs through a model to produce embeddings
    Encode(EncodeArgs),
    /// Launch the interactive TUI dashboard
    Tui,
}

#[derive(Parser)]
pub struct ModelsArgs {
    /// Filter by model family
    #[arg(short, long)]
    pub family: Option<ModelFamilyFilter>,

    /// Show full details for a specific model
    #[arg(short, long)]
    pub name: Option<String>,
}

#[derive(Clone, ValueEnum)]
pub enum ModelFamilyFilter {
    Ijepa,
    Vjepa,
}

#[derive(Parser)]
pub struct InspectArgs {
    /// Path to the model file (.safetensors or .onnx)
    #[arg(value_name = "FILE")]
    pub path: PathBuf,
}

#[derive(Parser)]
pub struct CheckpointArgs {
    /// Path to checkpoint file
    #[arg(value_name = "FILE")]
    pub path: PathBuf,

    /// Key-mapping preset to apply when loading
    #[arg(short, long, default_value = "ijepa")]
    pub keymap: KeymapPreset,

    /// Show per-tensor details
    #[arg(long)]
    pub verbose: bool,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum KeymapPreset {
    Ijepa,
    Vjepa,
    None,
}

#[derive(Parser)]
pub struct TrainArgs {
    /// Model architecture preset
    #[arg(short, long, default_value = "vit-base-16")]
    pub preset: ArchPreset,

    /// Total training steps
    #[arg(long, default_value_t = 1000)]
    pub steps: usize,

    /// Warmup steps
    #[arg(long, default_value_t = 100)]
    pub warmup: usize,

    /// Peak learning rate
    #[arg(long, default_value_t = 1e-3)]
    pub lr: f64,

    /// Batch size
    #[arg(long, default_value_t = 4)]
    pub batch_size: usize,

    /// Masking strategy
    #[arg(long, default_value = "block")]
    pub masking: MaskingChoice,

    /// Energy function
    #[arg(long, default_value = "l2")]
    pub energy: EnergyChoice,

    /// Collapse regularizer
    #[arg(long, default_value = "vicreg")]
    pub regularizer: RegularizerChoice,

    /// Regularization weight
    #[arg(long, default_value_t = 0.01)]
    pub reg_weight: f64,

    /// EMA base momentum
    #[arg(long, default_value_t = 0.996)]
    pub ema_momentum: f64,

    /// Log interval (steps)
    #[arg(long, default_value_t = 10)]
    pub log_interval: usize,

    /// Checkpoint interval (steps)
    #[arg(long, default_value_t = 100)]
    pub checkpoint_interval: usize,

    /// Checkpoint output directory
    #[arg(short, long, default_value = "./checkpoints")]
    pub output_dir: PathBuf,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum ArchPreset {
    VitBase16,
    VitSmall16,
    VitLarge16,
    VitHuge14,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum MaskingChoice {
    Block,
    MultiBlock,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum EnergyChoice {
    L2,
    Cosine,
    SmoothL1,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum RegularizerChoice {
    Vicreg,
    BarlowTwins,
}

#[derive(Parser)]
pub struct EncodeArgs {
    /// Path to checkpoint (.safetensors)
    #[arg(short, long)]
    pub model: PathBuf,

    /// Architecture preset (must match the checkpoint)
    #[arg(short, long, default_value = "vit-base-16")]
    pub preset: ArchPreset,

    /// Image dimensions (height)
    #[arg(long, default_value_t = 224)]
    pub height: usize,

    /// Image dimensions (width)
    #[arg(long, default_value_t = 224)]
    pub width: usize,

    /// Number of random samples to encode (demo mode)
    #[arg(long, default_value_t = 1)]
    pub num_samples: usize,
}
