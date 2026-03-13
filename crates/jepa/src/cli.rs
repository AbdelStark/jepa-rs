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
    Train(Box<TrainArgs>),
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
    #[arg(long, default_value_t = 1)]
    pub batch_size: usize,

    /// Optional safetensors dataset file containing an image tensor `[N, C, H, W]`
    #[arg(long, conflicts_with = "dataset_dir")]
    pub dataset: Option<PathBuf>,

    /// Tensor key to read from `--dataset`
    #[arg(long, default_value = "images")]
    pub dataset_key: String,

    /// Optional directory/tree of image files (`jpg`, `jpeg`, `png`, `webp`)
    #[arg(long, value_name = "PATH", conflicts_with = "dataset")]
    pub dataset_dir: Option<PathBuf>,

    /// Resize the shorter image side before center crop (`--dataset-dir` only)
    #[arg(long, value_name = "INT", requires = "dataset_dir")]
    pub resize: Option<usize>,

    /// Final square crop size after resize (`--dataset-dir` only)
    #[arg(long, value_name = "INT", requires = "dataset_dir")]
    pub crop_size: Option<usize>,

    /// Channel mean CSV for image-folder normalization (`--dataset-dir` only)
    #[arg(long, value_name = "CSV", requires = "dataset_dir")]
    pub mean: Option<String>,

    /// Channel std CSV for image-folder normalization (`--dataset-dir` only)
    #[arg(long, value_name = "CSV", requires = "dataset_dir")]
    pub std: Option<String>,

    /// Limit dataset-backed modes to the first N samples after discovery
    #[arg(long, value_name = "INT")]
    pub dataset_limit: Option<usize>,

    /// Shuffle dataset order at epoch boundaries for dataset-backed modes
    #[arg(long)]
    pub shuffle: bool,

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

#[derive(Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum ArchPreset {
    #[value(name = "vit-base-16", alias = "vit-base16")]
    VitBase16,
    #[value(name = "vit-small-16", alias = "vit-small16")]
    VitSmall16,
    #[value(name = "vit-large-16", alias = "vit-large16")]
    VitLarge16,
    #[value(name = "vit-huge-14", alias = "vit-huge14")]
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
    /// Path to model file (.onnx runtime, .safetensors burn-native load, others use preset demo mode)
    #[arg(short, long)]
    pub model: PathBuf,

    /// Architecture preset (used for burn-native models, ignored for .onnx)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_defaults_parse_with_documented_preset_name() {
        let cli = Cli::try_parse_from(["jepa", "train"]).expect("train defaults should parse");

        let Some(Command::Train(args)) = cli.command else {
            panic!("expected train subcommand");
        };
        assert_eq!(args.preset, ArchPreset::VitBase16);
    }

    #[test]
    fn encode_defaults_parse_with_documented_preset_name() {
        let cli = Cli::try_parse_from(["jepa", "encode", "--model", "model.onnx"])
            .expect("encode defaults should parse");

        let Some(Command::Encode(args)) = cli.command else {
            panic!("expected encode subcommand");
        };
        assert_eq!(args.preset, ArchPreset::VitBase16);
    }

    #[test]
    fn arch_preset_accepts_documented_and_legacy_aliases() {
        for preset in [
            "vit-base-16",
            "vit-base16",
            "vit-small-16",
            "vit-small16",
            "vit-large-16",
            "vit-large16",
            "vit-huge-14",
            "vit-huge14",
        ] {
            let cli = Cli::try_parse_from(["jepa", "train", "--preset", preset])
                .unwrap_or_else(|err| panic!("preset `{preset}` should parse: {err}"));

            let Some(Command::Train(_)) = cli.command else {
                panic!("expected train subcommand for preset `{preset}`");
            };
        }
    }

    #[test]
    fn train_accepts_optional_dataset_arguments() {
        let cli = Cli::try_parse_from([
            "jepa",
            "train",
            "--dataset",
            "train.safetensors",
            "--dataset-key",
            "images",
            "--dataset-limit",
            "4",
            "--shuffle",
        ])
        .expect("dataset-backed train flags should parse");

        let Some(Command::Train(args)) = cli.command else {
            panic!("expected train subcommand");
        };
        assert_eq!(
            args.dataset.as_deref(),
            Some(std::path::Path::new("train.safetensors"))
        );
        assert_eq!(args.dataset_key, "images");
        assert_eq!(args.dataset_limit, Some(4));
        assert!(args.shuffle);
    }

    #[test]
    fn train_accepts_image_folder_dataset_arguments() {
        let cli = Cli::try_parse_from([
            "jepa",
            "train",
            "--dataset-dir",
            "images/train",
            "--resize",
            "256",
            "--crop-size",
            "224",
            "--mean",
            "0.485,0.456,0.406",
            "--std",
            "0.229,0.224,0.225",
            "--shuffle",
        ])
        .expect("image-folder train flags should parse");

        let Some(Command::Train(args)) = cli.command else {
            panic!("expected train subcommand");
        };
        assert_eq!(
            args.dataset_dir.as_deref(),
            Some(std::path::Path::new("images/train"))
        );
        assert_eq!(args.resize, Some(256));
        assert_eq!(args.crop_size, Some(224));
        assert_eq!(args.mean.as_deref(), Some("0.485,0.456,0.406"));
        assert_eq!(args.std.as_deref(), Some("0.229,0.224,0.225"));
        assert!(args.shuffle);
    }

    #[test]
    fn train_rejects_multiple_dataset_sources() {
        let result = Cli::try_parse_from([
            "jepa",
            "train",
            "--dataset",
            "train.safetensors",
            "--dataset-dir",
            "images/train",
        ]);
        let err = match result {
            Ok(_) => panic!("multiple dataset sources should be rejected"),
            Err(err) => err,
        };

        let rendered = err.to_string();
        assert!(
            rendered.contains("--dataset") && rendered.contains("--dataset-dir"),
            "unexpected clap error: {rendered}"
        );
    }
}
