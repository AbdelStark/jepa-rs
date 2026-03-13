use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use burn::backend::Autodiff;
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn_ndarray::NdArray;
use image::imageops::{crop_imm, resize, FilterType};
use image::RgbImage;
use rand::seq::SliceRandom;

use jepa_core::{
    BarlowTwins, BlockMasking, CollapseRegularizer, CosineEnergy, Ema, EnergyFn, InputShape,
    L2Energy, MaskingStrategy, MultiBlockMasking, SmoothL1Energy, VICReg,
};
use jepa_train::{LrSchedule, WarmupCosineSchedule};
use jepa_vision::image::{IJepa, IJepaConfig, TransformerPredictorConfig};
use jepa_vision::vit::VitConfig;

use crate::cli::{ArchPreset, EnergyChoice, MaskingChoice, RegularizerChoice, TrainArgs};

type B = Autodiff<NdArray<f32>>;
const DEVICE: burn_ndarray::NdArrayDevice = burn_ndarray::NdArrayDevice::Cpu;
const DEFAULT_RGB_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DEFAULT_RGB_STD: [f32; 3] = [0.229, 0.224, 0.225];
const SUPPORTED_IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrainSourceKind {
    Synthetic,
    Safetensors,
    ImageFolder,
}

#[derive(Debug, Clone)]
pub(crate) struct TrainRunSummary {
    pub source_kind: TrainSourceKind,
    pub source_count: Option<usize>,
    pub source_description: String,
    pub preset: ArchPreset,
    pub embed_dim: usize,
    pub patch_size: (usize, usize),
    pub image_size: (usize, usize),
    pub num_patches: usize,
    pub steps: usize,
    pub warmup: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub masking: MaskingChoice,
    pub energy: EnergyChoice,
    pub regularizer: RegularizerChoice,
    pub reg_weight: f64,
    pub ema_momentum: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct TrainStepMetrics {
    pub step: usize,
    pub total_steps: usize,
    pub energy: f64,
    pub regularization: f64,
    pub total_loss: f64,
    pub learning_rate: f64,
}

pub(crate) trait TrainReporter {
    fn on_run_started(&mut self, _summary: &TrainRunSummary) {}

    fn on_step(&mut self, _metrics: &TrainStepMetrics) {}

    fn on_run_complete(&mut self, _summary: &TrainRunSummary) {}
}

struct TerminalReporter {
    log_interval: usize,
}

impl TerminalReporter {
    fn new(log_interval: usize) -> Self {
        Self { log_interval }
    }
}

impl TrainReporter for TerminalReporter {
    fn on_run_started(&mut self, summary: &TrainRunSummary) {
        println!();
        print_banner(summary);

        let (patch_h, patch_w) = summary.patch_size;
        let (image_h, image_w) = summary.image_size;

        println!("  Architecture:   {:?}", summary.preset);
        println!("  Embed dim:      {}", summary.embed_dim);
        println!("  Patch size:     {patch_h}x{patch_w}");
        println!("  Image size:     {image_h}x{image_w}");
        println!("  Num patches:    {}", summary.num_patches);
        println!("  Steps:          {}", summary.steps);
        println!("  Warmup:         {}", summary.warmup);
        println!("  LR:             {}", summary.learning_rate);
        println!("  Batch size:     {}", summary.batch_size);
        println!("  Masking:        {:?}", summary.masking);
        println!("  Energy:         {:?}", summary.energy);
        println!("  Regularizer:    {:?}", summary.regularizer);
        println!("  Reg weight:     {}", summary.reg_weight);
        println!("  EMA momentum:   {}", summary.ema_momentum);
        println!("  Data source:    {}", summary.source_description);
        println!();
        println!("  ┌────────┬──────────────┬──────────────┬──────────────┬──────────┐");
        println!("  │  Step  │    Energy    │     Reg      │  Total Loss  │    LR    │");
        println!("  ├────────┼──────────────┼──────────────┼──────────────┼──────────┤");
    }

    fn on_step(&mut self, metrics: &TrainStepMetrics) {
        if metrics.step % self.log_interval == 0 || metrics.step + 1 == metrics.total_steps {
            println!(
                "  │ {:>5}  │ {:>12.6} │ {:>12.6} │ {:>12.6} │ {:>8.2e} │",
                metrics.step,
                metrics.energy,
                metrics.regularization,
                metrics.total_loss,
                metrics.learning_rate,
            );
        }
    }

    fn on_run_complete(&mut self, _summary: &TrainRunSummary) {
        println!("  └────────┴──────────────┴──────────────┴──────────────┴──────────┘");
        println!();
        println!("  Training complete.");
        println!();
    }
}

pub fn run(args: TrainArgs) -> Result<()> {
    let mut reporter = TerminalReporter::new(args.log_interval);
    run_with_reporter(args, &mut reporter)
}

pub(crate) fn run_with_reporter<R>(args: TrainArgs, reporter: &mut R) -> Result<()>
where
    R: TrainReporter,
{
    let vit_config = match args.preset {
        ArchPreset::VitBase16 => VitConfig::vit_base_patch16(),
        ArchPreset::VitSmall16 => VitConfig::vit_small_patch16(),
        ArchPreset::VitLarge16 => VitConfig::vit_large_patch16(),
        ArchPreset::VitHuge14 => VitConfig::vit_huge_patch14(),
    };

    let mut batch_source = BatchSource::from_args(&args, &vit_config)?;

    let embed_dim = vit_config.embed_dim;
    let mask_shape = training_input_shape(&vit_config);
    let summary = TrainRunSummary {
        source_kind: batch_source.kind(),
        source_count: batch_source.sample_count(),
        source_description: batch_source.describe(),
        preset: args.preset.clone(),
        embed_dim,
        patch_size: vit_config.patch_size,
        image_size: (vit_config.image_height, vit_config.image_width),
        num_patches: mask_shape.total_tokens(),
        steps: args.steps,
        warmup: args.warmup,
        learning_rate: args.lr,
        batch_size: args.batch_size,
        masking: args.masking.clone(),
        energy: args.energy.clone(),
        regularizer: args.regularizer.clone(),
        reg_weight: args.reg_weight,
        ema_momentum: args.ema_momentum,
    };
    reporter.on_run_started(&summary);

    let predictor_config = TransformerPredictorConfig {
        encoder_embed_dim: embed_dim,
        predictor_embed_dim: embed_dim / 4,
        num_layers: 6,
        num_heads: vit_config.num_heads,
        max_target_len: summary.num_patches,
    };

    let mut model: IJepa<B> = IJepaConfig {
        encoder: vit_config.clone(),
        predictor: predictor_config,
    }
    .init(&DEVICE);
    model.target_encoder = model.target_encoder.no_grad();

    let mut optimizer = AdamWConfig::new().init::<B, IJepa<B>>();
    let ema = Ema::with_cosine_schedule(args.ema_momentum, args.steps);

    let block_masking = BlockMasking {
        num_targets: 4,
        target_scale: (0.15, 0.4),
        target_aspect_ratio: (0.75, 1.5),
    };
    let multi_block_masking = MultiBlockMasking {
        mask_ratio: 0.4,
        num_blocks: 4,
    };

    model = match (&args.energy, &args.regularizer) {
        (EnergyChoice::L2, RegularizerChoice::Vicreg) => run_loop(
            &args,
            model,
            &mut optimizer,
            &ema,
            &L2Energy,
            &VICReg::default(),
            &args.masking,
            &block_masking,
            &multi_block_masking,
            &mask_shape,
            &mut batch_source,
            reporter,
        )?,
        (EnergyChoice::L2, RegularizerChoice::BarlowTwins) => run_loop(
            &args,
            model,
            &mut optimizer,
            &ema,
            &L2Energy,
            &BarlowTwins::default(),
            &args.masking,
            &block_masking,
            &multi_block_masking,
            &mask_shape,
            &mut batch_source,
            reporter,
        )?,
        (EnergyChoice::Cosine, RegularizerChoice::Vicreg) => run_loop(
            &args,
            model,
            &mut optimizer,
            &ema,
            &CosineEnergy,
            &VICReg::default(),
            &args.masking,
            &block_masking,
            &multi_block_masking,
            &mask_shape,
            &mut batch_source,
            reporter,
        )?,
        (EnergyChoice::Cosine, RegularizerChoice::BarlowTwins) => run_loop(
            &args,
            model,
            &mut optimizer,
            &ema,
            &CosineEnergy,
            &BarlowTwins::default(),
            &args.masking,
            &block_masking,
            &multi_block_masking,
            &mask_shape,
            &mut batch_source,
            reporter,
        )?,
        (EnergyChoice::SmoothL1, RegularizerChoice::Vicreg) => run_loop(
            &args,
            model,
            &mut optimizer,
            &ema,
            &SmoothL1Energy::new(1.0),
            &VICReg::default(),
            &args.masking,
            &block_masking,
            &multi_block_masking,
            &mask_shape,
            &mut batch_source,
            reporter,
        )?,
        (EnergyChoice::SmoothL1, RegularizerChoice::BarlowTwins) => run_loop(
            &args,
            model,
            &mut optimizer,
            &ema,
            &SmoothL1Energy::new(1.0),
            &BarlowTwins::default(),
            &args.masking,
            &block_masking,
            &multi_block_masking,
            &mask_shape,
            &mut batch_source,
            reporter,
        )?,
    };

    // Keep the final model alive until the end of the command.
    let _ = model;
    reporter.on_run_complete(&summary);

    Ok(())
}

fn training_input_shape(vit_config: &VitConfig) -> InputShape {
    let (patch_h, patch_w) = vit_config.patch_size;
    InputShape::Image {
        height: vit_config.image_height / patch_h,
        width: vit_config.image_width / patch_w,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_loop<EF, CR, O>(
    args: &TrainArgs,
    mut model: IJepa<B>,
    optimizer: &mut O,
    ema: &Ema,
    energy_fn: &EF,
    regularizer: &CR,
    masking_choice: &MaskingChoice,
    block_masking: &BlockMasking,
    multi_block_masking: &MultiBlockMasking,
    input_shape: &InputShape,
    batch_source: &mut BatchSource,
    reporter: &mut impl TrainReporter,
) -> Result<IJepa<B>>
where
    EF: EnergyFn<B>,
    CR: CollapseRegularizer<B>,
    O: Optimizer<IJepa<B>, B>,
{
    let mut rng = rand::rng();
    let lr_schedule = WarmupCosineSchedule::new(args.lr, args.warmup, args.steps);

    for step in 0..args.steps {
        let lr = lr_schedule.get_lr(step);
        let input = batch_source
            .next_batch(args.batch_size, &DEVICE)
            .with_context(|| format!("failed to prepare batch for step {step}"))?;

        let output = match masking_choice {
            MaskingChoice::Block => {
                let mask = block_masking.generate_mask(input_shape, &mut rng);
                model.try_forward_step_strict(&input, mask, energy_fn, regularizer, args.reg_weight)
            }
            MaskingChoice::MultiBlock => {
                let mask = multi_block_masking.generate_mask(input_shape, &mut rng);
                model.try_forward_step_strict(&input, mask, energy_fn, regularizer, args.reg_weight)
            }
        }
        .with_context(|| format!("training step {step} failed"))?;

        reporter.on_step(&TrainStepMetrics {
            step,
            total_steps: args.steps,
            energy: f64::from(output.energy.value.clone().into_scalar()),
            regularization: f64::from(output.regularization.clone().into_scalar()),
            total_loss: f64::from(output.total_loss.clone().into_scalar()),
            learning_rate: lr,
        });

        let grads = GradientsParams::from_grads(output.total_loss.backward(), &model);
        model = optimizer.step(lr, model, grads);
        model.target_encoder = model
            .target_encoder
            .clone()
            .ema_update_from(&model.context_encoder, ema, step)
            .no_grad();
    }

    Ok(model)
}

fn print_banner(summary: &TrainRunSummary) {
    println!("  ╔══════════════════════════════════════════════════════════════╗");
    match summary.source_kind {
        TrainSourceKind::Synthetic => {
            println!("  ║            JEPA Training — Synthetic Demo                  ║");
            println!("  ║                                                            ║");
            println!("  ║  Optimizer and EMA are active on synthetic random data.    ║");
            println!("  ║  Pass --dataset-dir or --dataset for real data.            ║");
        }
        TrainSourceKind::Safetensors => {
            println!("  ║          JEPA Training — Safetensors Dataset               ║");
            println!("  ║                                                            ║");
            println!("  ║  Strict masking, AdamW, and EMA are active.                ║");
            println!(
                "  ║  Loaded {:>5} image tensor(s) from safetensors dataset.     ║",
                summary.source_count.unwrap_or(0)
            );
        }
        TrainSourceKind::ImageFolder => {
            println!("  ║           JEPA Training — Image Folder Mode                ║");
            println!("  ║                                                            ║");
            println!("  ║  Strict masking, AdamW, and EMA are active.                ║");
            println!(
                "  ║  Loaded {:>5} image file(s) with deterministic prep.        ║",
                summary.source_count.unwrap_or(0)
            );
        }
    }
    println!("  ╚══════════════════════════════════════════════════════════════╝");
    println!();
}

#[derive(Debug, Clone, PartialEq)]
struct NormalizationStats {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl NormalizationStats {
    fn from_args(args: &TrainArgs, expected_channels: usize) -> Result<Self> {
        let mean = match args.mean.as_deref() {
            Some(csv) => Self::expand(parse_csv_f32(csv, "--mean")?, expected_channels, "--mean")?,
            None => default_mean(expected_channels),
        };
        let std = match args.std.as_deref() {
            Some(csv) => Self::expand(parse_csv_f32(csv, "--std")?, expected_channels, "--std")?,
            None => default_std(expected_channels),
        };

        for (index, value) in std.iter().enumerate() {
            if value.abs() <= f32::EPSILON {
                bail!("--std channel {index} must be non-zero");
            }
        }

        Ok(Self { mean, std })
    }

    fn expand(values: Vec<f32>, expected_channels: usize, label: &str) -> Result<Vec<f32>> {
        match values.len() {
            1 => Ok(vec![values[0]; expected_channels]),
            len if len == expected_channels => Ok(values),
            len => bail!(
                "{label} must contain either 1 value or {expected_channels} values, got {len}"
            ),
        }
    }
}

#[derive(Debug, Clone)]
struct ImageFolderOptions {
    resize: usize,
    crop_size: usize,
    normalization: NormalizationStats,
    shuffle: bool,
    dataset_limit: Option<usize>,
}

impl ImageFolderOptions {
    fn from_args(args: &TrainArgs, vit_config: &VitConfig) -> Result<Self> {
        if vit_config.in_channels != 3 {
            bail!(
                "image-folder datasets currently require an RGB model with 3 input channels, got {}",
                vit_config.in_channels
            );
        }

        if vit_config.image_height != vit_config.image_width {
            bail!(
                "image-folder datasets currently require a square preset image size, got {}x{}",
                vit_config.image_height,
                vit_config.image_width
            );
        }

        let crop_size = args.crop_size.unwrap_or(vit_config.image_height);
        if crop_size == 0 {
            bail!("--crop-size must be positive");
        }
        if crop_size != vit_config.image_height {
            bail!(
                "image-folder crop size must match the preset image size {} for {:?}, got {}",
                vit_config.image_height,
                args.preset,
                crop_size
            );
        }

        let resize = args.resize.unwrap_or(crop_size);
        if resize == 0 {
            bail!("--resize must be positive");
        }
        if resize < crop_size {
            bail!("--resize ({resize}) must be >= --crop-size ({crop_size})");
        }

        Ok(Self {
            resize,
            crop_size,
            normalization: NormalizationStats::from_args(args, vit_config.in_channels)?,
            shuffle: args.shuffle,
            dataset_limit: args.dataset_limit,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct TensorDatasetOptions {
    expected_channels: usize,
    expected_height: usize,
    expected_width: usize,
    shuffle: bool,
    dataset_limit: Option<usize>,
}

enum BatchSource {
    Synthetic {
        channels: usize,
        height: usize,
        width: usize,
    },
    Safetensors(ImageTensorDataset),
    ImageFolder(ImageFolderDataset),
}

impl BatchSource {
    fn from_args(args: &TrainArgs, vit_config: &VitConfig) -> Result<Self> {
        match (&args.dataset, &args.dataset_dir) {
            (Some(path), None) => Ok(Self::Safetensors(ImageTensorDataset::from_safetensors(
                path,
                &args.dataset_key,
                TensorDatasetOptions {
                    expected_channels: vit_config.in_channels,
                    expected_height: vit_config.image_height,
                    expected_width: vit_config.image_width,
                    shuffle: args.shuffle,
                    dataset_limit: args.dataset_limit,
                },
            )?)),
            (None, Some(dir)) => Ok(Self::ImageFolder(ImageFolderDataset::from_directory(
                dir,
                vit_config.in_channels,
                vit_config.image_height,
                vit_config.image_width,
                ImageFolderOptions::from_args(args, vit_config)?,
            )?)),
            (None, None) => Ok(Self::Synthetic {
                channels: vit_config.in_channels,
                height: vit_config.image_height,
                width: vit_config.image_width,
            }),
            (Some(_), Some(_)) => bail!("--dataset and --dataset-dir are mutually exclusive"),
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Synthetic { .. } => "synthetic random tensors".to_string(),
            Self::Safetensors(dataset) => format!(
                "{}:{} [{} samples{}]",
                dataset.path.display(),
                dataset.tensor_key,
                dataset.num_samples,
                if dataset.shuffle { ", shuffled" } else { "" }
            ),
            Self::ImageFolder(dataset) => format!(
                "{} [{} files, resize {}, crop {}{}]",
                dataset.root.display(),
                dataset.num_samples(),
                dataset.resize,
                dataset.crop_size,
                if dataset.shuffle { ", shuffled" } else { "" }
            ),
        }
    }

    fn kind(&self) -> TrainSourceKind {
        match self {
            Self::Synthetic { .. } => TrainSourceKind::Synthetic,
            Self::Safetensors(_) => TrainSourceKind::Safetensors,
            Self::ImageFolder(_) => TrainSourceKind::ImageFolder,
        }
    }

    fn sample_count(&self) -> Option<usize> {
        match self {
            Self::Synthetic { .. } => None,
            Self::Safetensors(dataset) => Some(dataset.num_samples),
            Self::ImageFolder(dataset) => Some(dataset.num_samples()),
        }
    }

    fn next_batch(
        &mut self,
        batch_size: usize,
        device: &<B as Backend>::Device,
    ) -> Result<Tensor<B, 4>> {
        match self {
            Self::Synthetic {
                channels,
                height,
                width,
            } => Ok(Tensor::random(
                [batch_size, *channels, *height, *width],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                device,
            )),
            Self::Safetensors(dataset) => dataset.next_batch(batch_size, device),
            Self::ImageFolder(dataset) => dataset.next_batch(batch_size, device),
        }
    }
}

struct ImageTensorDataset {
    path: PathBuf,
    tensor_key: String,
    data: Vec<f32>,
    num_samples: usize,
    channels: usize,
    height: usize,
    width: usize,
    order: Vec<usize>,
    cursor: usize,
    shuffle: bool,
}

impl ImageTensorDataset {
    fn from_safetensors(
        path: &Path,
        tensor_key: &str,
        options: TensorDatasetOptions,
    ) -> Result<Self> {
        let checkpoint = jepa_compat::safetensors::load_raw_checkpoint(path)
            .with_context(|| format!("failed to load dataset {}", path.display()))?;
        let tensor = checkpoint.get(tensor_key).with_context(|| {
            format!(
                "dataset tensor `{tensor_key}` not found in {}",
                path.display()
            )
        })?;

        Self::from_loaded_tensor(
            path.to_path_buf(),
            tensor_key.to_string(),
            tensor.to_tensor_data(),
            options,
        )
    }

    fn from_loaded_tensor(
        path: PathBuf,
        tensor_key: String,
        tensor: TensorData,
        options: TensorDatasetOptions,
    ) -> Result<Self> {
        let [num_samples, channels, height, width] = tensor.shape.as_slice() else {
            bail!(
                "dataset tensor `{tensor_key}` in {} must have shape [N, C, H, W], got {:?}",
                path.display(),
                tensor.shape
            );
        };

        if *num_samples == 0 {
            bail!(
                "dataset tensor `{tensor_key}` in {} is empty",
                path.display()
            );
        }

        if *channels != options.expected_channels
            || *height != options.expected_height
            || *width != options.expected_width
        {
            bail!(
                "dataset tensor `{tensor_key}` in {} must match [{} , {}, {}, {}], got {:?}",
                path.display(),
                "N",
                options.expected_channels,
                options.expected_height,
                options.expected_width,
                tensor.shape
            );
        }

        if let Some(limit) = options.dataset_limit {
            if limit == 0 {
                bail!("--dataset-limit must be positive");
            }
        }

        let num_samples = options
            .dataset_limit
            .map(|limit| limit.min(*num_samples))
            .unwrap_or(*num_samples);
        let data = tensor.to_vec::<f32>().map_err(|err| {
            anyhow::anyhow!("failed to decode dataset tensor `{tensor_key}`: {err}")
        })?;
        let mut order: Vec<usize> = (0..num_samples).collect();
        maybe_shuffle(&mut order, options.shuffle);

        Ok(Self {
            path,
            tensor_key,
            data,
            num_samples,
            channels: *channels,
            height: *height,
            width: *width,
            order,
            cursor: 0,
            shuffle: options.shuffle,
        })
    }

    fn next_batch(
        &mut self,
        batch_size: usize,
        device: &<B as Backend>::Device,
    ) -> Result<Tensor<B, 4>> {
        if batch_size == 0 {
            bail!("batch_size must be positive");
        }

        let sample_len = self.channels * self.height * self.width;
        let mut batch = Vec::with_capacity(batch_size * sample_len);

        for _ in 0..batch_size {
            if self.cursor >= self.order.len() {
                maybe_shuffle(&mut self.order, self.shuffle);
                self.cursor = 0;
            }
            let sample_index = self.order[self.cursor];
            self.cursor += 1;

            let start = sample_index * sample_len;
            let end = start + sample_len;
            batch.extend_from_slice(&self.data[start..end]);
        }

        Ok(Tensor::from_floats(
            TensorData::new(batch, [batch_size, self.channels, self.height, self.width]),
            device,
        ))
    }
}

struct ImageFolderDataset {
    root: PathBuf,
    file_paths: Vec<PathBuf>,
    channels: usize,
    height: usize,
    width: usize,
    resize: usize,
    crop_size: usize,
    normalization: NormalizationStats,
    order: Vec<usize>,
    cursor: usize,
    shuffle: bool,
}

impl ImageFolderDataset {
    fn from_directory(
        root: &Path,
        expected_channels: usize,
        expected_height: usize,
        expected_width: usize,
        options: ImageFolderOptions,
    ) -> Result<Self> {
        if expected_channels != 3 {
            bail!(
                "image-folder datasets currently produce RGB tensors with 3 channels, got expected channel count {}",
                expected_channels
            );
        }
        if expected_height != options.crop_size || expected_width != options.crop_size {
            bail!(
                "image-folder crop size {} must match model input dimensions {}x{}",
                options.crop_size,
                expected_height,
                expected_width
            );
        }

        let mut file_paths = collect_image_files(root)?;
        if file_paths.is_empty() {
            bail!(
                "no supported image files found under {} (expected one of: {})",
                root.display(),
                SUPPORTED_IMAGE_EXTENSIONS.join(", ")
            );
        }

        if let Some(limit) = options.dataset_limit {
            if limit == 0 {
                bail!("--dataset-limit must be positive");
            }
            file_paths.truncate(limit.min(file_paths.len()));
        }

        let mut order: Vec<usize> = (0..file_paths.len()).collect();
        maybe_shuffle(&mut order, options.shuffle);

        Ok(Self {
            root: root.to_path_buf(),
            file_paths,
            channels: expected_channels,
            height: expected_height,
            width: expected_width,
            resize: options.resize,
            crop_size: options.crop_size,
            normalization: options.normalization,
            order,
            cursor: 0,
            shuffle: options.shuffle,
        })
    }

    fn num_samples(&self) -> usize {
        self.file_paths.len()
    }

    fn next_batch(
        &mut self,
        batch_size: usize,
        device: &<B as Backend>::Device,
    ) -> Result<Tensor<B, 4>> {
        if batch_size == 0 {
            bail!("batch_size must be positive");
        }

        let sample_len = self.channels * self.height * self.width;
        let mut batch = Vec::with_capacity(batch_size * sample_len);

        for _ in 0..batch_size {
            if self.cursor >= self.order.len() {
                maybe_shuffle(&mut self.order, self.shuffle);
                self.cursor = 0;
            }

            let sample_index = self.order[self.cursor];
            self.cursor += 1;
            let sample = self.load_sample(sample_index)?;
            batch.extend_from_slice(&sample);
        }

        Ok(Tensor::from_floats(
            TensorData::new(batch, [batch_size, self.channels, self.height, self.width]),
            device,
        ))
    }

    fn load_sample(&self, index: usize) -> Result<Vec<f32>> {
        let path = &self.file_paths[index];
        let image = image::open(path)
            .with_context(|| format!("failed to decode image {}", path.display()))?;
        let rgb = image.to_rgb8();
        let resized = resize_shorter_side(&rgb, self.resize);

        if resized.width() < self.crop_size as u32 || resized.height() < self.crop_size as u32 {
            bail!(
                "image {} resized to {}x{} which is smaller than crop size {}",
                path.display(),
                resized.width(),
                resized.height(),
                self.crop_size
            );
        }

        let crop = self.crop_size as u32;
        let left = (resized.width() - crop) / 2;
        let top = (resized.height() - crop) / 2;
        let cropped = crop_imm(&resized, left, top, crop, crop).to_image();

        Ok(rgb_image_to_chw(&cropped, &self.normalization))
    }
}

fn maybe_shuffle(order: &mut [usize], shuffle: bool) {
    if shuffle {
        order.shuffle(&mut rand::rng());
    }
}

fn default_mean(expected_channels: usize) -> Vec<f32> {
    match expected_channels {
        3 => DEFAULT_RGB_MEAN.to_vec(),
        1 => vec![0.5],
        _ => vec![0.0; expected_channels],
    }
}

fn default_std(expected_channels: usize) -> Vec<f32> {
    match expected_channels {
        3 => DEFAULT_RGB_STD.to_vec(),
        1 => vec![0.5],
        _ => vec![1.0; expected_channels],
    }
}

fn parse_csv_f32(csv: &str, label: &str) -> Result<Vec<f32>> {
    let values = csv
        .split(',')
        .map(str::trim)
        .map(|value| {
            if value.is_empty() {
                bail!("{label} contains an empty value");
            }
            value
                .parse::<f32>()
                .with_context(|| format!("failed to parse {label} value `{value}`"))
        })
        .collect::<Result<Vec<_>>>()?;

    if values.is_empty() {
        bail!("{label} must contain at least one value");
    }

    Ok(values)
}

fn collect_image_files(root: &Path) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        bail!("dataset directory {} does not exist", root.display());
    }
    if !root.is_dir() {
        bail!("dataset directory {} is not a directory", root.display());
    }

    let mut files = Vec::new();
    collect_image_files_recursive(root, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_image_files_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("failed to read dataset directory {}", dir.display()))?
    {
        let entry = entry.with_context(|| format!("failed to read entry in {}", dir.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect {}", entry.path().display()))?;
        let path = entry.path();

        if file_type.is_dir() {
            collect_image_files_recursive(&path, files)?;
        } else if file_type.is_file() && is_supported_image_file(&path) {
            files.push(path);
        }
    }

    Ok(())
}

fn is_supported_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            let extension = extension.to_ascii_lowercase();
            SUPPORTED_IMAGE_EXTENSIONS.contains(&extension.as_str())
        })
        .unwrap_or(false)
}

fn resize_shorter_side(image: &RgbImage, shorter_side: usize) -> RgbImage {
    let width = image.width() as usize;
    let height = image.height() as usize;

    let (new_width, new_height) = if width <= height {
        (shorter_side, scaled_dimension(height, shorter_side, width))
    } else {
        (scaled_dimension(width, shorter_side, height), shorter_side)
    };

    resize(
        image,
        new_width as u32,
        new_height as u32,
        FilterType::Triangle,
    )
}

fn scaled_dimension(
    original: usize,
    target_shorter_side: usize,
    original_shorter_side: usize,
) -> usize {
    let scaled = (original as u64 * target_shorter_side as u64 + original_shorter_side as u64 / 2)
        / original_shorter_side as u64;
    scaled.max(1) as usize
}

fn rgb_image_to_chw(image: &RgbImage, normalization: &NormalizationStats) -> Vec<f32> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let channels = normalization.mean.len();
    let mut data = vec![0.0f32; channels * height * width];

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x as u32, y as u32).0;
            for (channel, component) in pixel.iter().enumerate().take(channels) {
                let value = f32::from(*component) / 255.0;
                let normalized = (value - normalization.mean[channel]) / normalization.std[channel];
                let index = channel * height * width + y * width + x;
                data[index] = normalized;
            }
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};
    use rand::{rngs::StdRng, SeedableRng};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn training_input_shape_uses_patch_grid_dimensions() {
        let vit_config = VitConfig::vit_small_patch16();
        let shape = training_input_shape(&vit_config);

        assert!(matches!(
            shape,
            InputShape::Image {
                height: 14,
                width: 14,
            }
        ));
        assert_eq!(shape.total_tokens(), 196);

        let masking = BlockMasking {
            num_targets: 4,
            target_scale: (0.15, 0.4),
            target_aspect_ratio: (0.75, 1.5),
        };
        let mut rng = StdRng::seed_from_u64(7);

        for _ in 0..8 {
            let mask = masking.generate_mask(&shape, &mut rng);
            assert!(mask
                .context_indices
                .iter()
                .chain(mask.target_indices.iter())
                .all(|&index| index < shape.total_tokens()));
        }
    }

    #[test]
    fn strict_training_smoke_step_runs_with_tiny_vit() {
        let vit_config = VitConfig::tiny_test();
        let input_shape = training_input_shape(&vit_config);
        let mut model: IJepa<B> = IJepaConfig {
            encoder: vit_config.clone(),
            predictor: TransformerPredictorConfig {
                encoder_embed_dim: vit_config.embed_dim,
                predictor_embed_dim: 16,
                num_layers: 1,
                num_heads: vit_config.num_heads,
                max_target_len: input_shape.total_tokens(),
            },
        }
        .init(&DEVICE);
        model.target_encoder = model.target_encoder.no_grad();

        let input = Tensor::random(
            [
                2,
                vit_config.in_channels,
                vit_config.image_height,
                vit_config.image_width,
            ],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &DEVICE,
        );

        let masking = BlockMasking {
            num_targets: 2,
            target_scale: (0.15, 0.4),
            target_aspect_ratio: (0.75, 1.5),
        };
        let mut rng = StdRng::seed_from_u64(123);
        let mask = masking.generate_mask(&input_shape, &mut rng);

        let output = model
            .try_forward_step_strict(&input, mask.clone(), &L2Energy, &VICReg::default(), 0.01)
            .expect("strict synthetic training step should succeed for CLI configs");

        assert_eq!(output.mask.total_tokens, input_shape.total_tokens());
        assert_eq!(output.predicted.seq_len(), mask.target_indices.len());
        assert_eq!(output.target.seq_len(), mask.target_indices.len());
    }

    #[test]
    fn dataset_batches_match_requested_shape() {
        let dataset = ImageTensorDataset::from_loaded_tensor(
            PathBuf::from("test.safetensors"),
            "images".to_string(),
            TensorData::new(vec![0.5f32; 3 * 8 * 8], [3, 1, 8, 8]),
            TensorDatasetOptions {
                expected_channels: 1,
                expected_height: 8,
                expected_width: 8,
                shuffle: false,
                dataset_limit: None,
            },
        )
        .expect("test dataset should load");

        let mut batch_source = BatchSource::Safetensors(dataset);
        let batch = batch_source
            .next_batch(2, &DEVICE)
            .expect("dataset batch should materialize");
        assert_eq!(batch.dims(), [2, 1, 8, 8]);
    }

    #[test]
    fn image_folder_dataset_decodes_to_normalized_chw_rgb() {
        let root = make_temp_dir("image-folder-dataset");
        let nested = root.join("class_a");
        std::fs::create_dir_all(&nested).expect("test image directory should exist");

        let image_path = nested.join("sample.png");
        let image = GrayImage::from_fn(2, 2, |x, y| match (x, y) {
            (0, 0) => Luma([0u8]),
            (1, 0) => Luma([64u8]),
            (0, 1) => Luma([128u8]),
            (1, 1) => Luma([255u8]),
            _ => unreachable!("2x2 image coordinates should stay in range"),
        });
        image
            .save(&image_path)
            .expect("fixture image should save successfully");

        let mut dataset = ImageFolderDataset::from_directory(
            &root,
            3,
            2,
            2,
            ImageFolderOptions {
                resize: 2,
                crop_size: 2,
                normalization: NormalizationStats {
                    mean: vec![0.0, 0.0, 0.0],
                    std: vec![1.0, 1.0, 1.0],
                },
                shuffle: false,
                dataset_limit: Some(1),
            },
        )
        .expect("image-folder dataset should load");

        assert_eq!(dataset.num_samples(), 1);

        let batch = dataset
            .next_batch(1, &DEVICE)
            .expect("image-folder batch should materialize");
        assert_eq!(batch.dims(), [1, 3, 2, 2]);

        let data = batch
            .to_data()
            .to_vec::<f32>()
            .expect("batch tensor should decode to f32 values");
        let expected = [
            0.0,
            64.0 / 255.0,
            128.0 / 255.0,
            1.0,
            0.0,
            64.0 / 255.0,
            128.0 / 255.0,
            1.0,
            0.0,
            64.0 / 255.0,
            128.0 / 255.0,
            1.0,
        ];

        for (actual, expected) in data.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "expected {expected}, got {actual}"
            );
        }

        std::fs::remove_dir_all(&root).expect("temporary test directory should be removed");
    }

    #[test]
    fn optimizer_step_updates_tiny_model() {
        let vit_config = VitConfig::tiny_test();
        let input_shape = training_input_shape(&vit_config);
        let mut model: IJepa<B> = IJepaConfig {
            encoder: vit_config.clone(),
            predictor: TransformerPredictorConfig {
                encoder_embed_dim: vit_config.embed_dim,
                predictor_embed_dim: 16,
                num_layers: 1,
                num_heads: vit_config.num_heads,
                max_target_len: input_shape.total_tokens(),
            },
        }
        .init(&DEVICE);
        model.target_encoder = model.target_encoder.no_grad();

        let input = Tensor::random(
            [2, 1, 8, 8],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &DEVICE,
        );
        let before = model.context_encoder.forward(&input);

        let masking = BlockMasking {
            num_targets: 2,
            target_scale: (0.15, 0.4),
            target_aspect_ratio: (0.75, 1.5),
        };
        let mut rng = StdRng::seed_from_u64(999);
        let mask = masking.generate_mask(&input_shape, &mut rng);

        let output = model
            .try_forward_step_strict(&input, mask, &L2Energy, &VICReg::default(), 0.01)
            .expect("strict training step should succeed");

        let grads = GradientsParams::from_grads(output.total_loss.backward(), &model);
        let mut optimizer = AdamWConfig::new().init::<B, IJepa<B>>();
        model = optimizer.step(1e-2, model, grads);
        model.target_encoder = model
            .target_encoder
            .clone()
            .ema_update_from(&model.context_encoder, &Ema::new(0.5), 0)
            .no_grad();

        let after = model.context_encoder.forward(&input);
        let delta: f32 = (after.embeddings - before.embeddings)
            .abs()
            .sum()
            .into_scalar();
        assert!(delta > 0.0, "optimizer step should update the model");
    }

    fn make_temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("jepa-rs-{label}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&path).expect("temporary test directory should be created");
        path
    }

    #[test]
    fn is_supported_image_file_accepts_common_extensions() {
        assert!(is_supported_image_file(Path::new("photo.jpg")));
        assert!(is_supported_image_file(Path::new("photo.jpeg")));
        assert!(is_supported_image_file(Path::new("photo.png")));
        assert!(is_supported_image_file(Path::new("photo.webp")));
    }

    #[test]
    fn is_supported_image_file_rejects_non_image() {
        assert!(!is_supported_image_file(Path::new("data.txt")));
        assert!(!is_supported_image_file(Path::new("model.onnx")));
        assert!(!is_supported_image_file(Path::new("no_extension")));
    }

    #[test]
    fn is_supported_image_file_case_insensitive() {
        assert!(is_supported_image_file(Path::new("photo.PNG")));
        assert!(is_supported_image_file(Path::new("photo.Jpg")));
    }

    #[test]
    fn scaled_dimension_preserves_aspect_ratio() {
        // 200x100 image, target shorter side = 50
        // width should scale to 100
        assert_eq!(scaled_dimension(200, 50, 100), 100);
    }

    #[test]
    fn scaled_dimension_minimum_is_one() {
        assert_eq!(scaled_dimension(1, 1, 1000), 1);
    }

    #[test]
    fn scaled_dimension_square() {
        assert_eq!(scaled_dimension(100, 50, 100), 50);
    }

    #[test]
    fn default_mean_rgb() {
        let mean = default_mean(3);
        assert_eq!(mean.len(), 3);
        assert!((mean[0] - 0.485).abs() < 1e-6);
    }

    #[test]
    fn default_mean_grayscale() {
        let mean = default_mean(1);
        assert_eq!(mean, vec![0.5]);
    }

    #[test]
    fn default_mean_other_channels() {
        let mean = default_mean(5);
        assert_eq!(mean, vec![0.0; 5]);
    }

    #[test]
    fn default_std_rgb() {
        let std_vals = default_std(3);
        assert_eq!(std_vals.len(), 3);
        assert!((std_vals[0] - 0.229).abs() < 1e-6);
    }

    #[test]
    fn default_std_grayscale() {
        assert_eq!(default_std(1), vec![0.5]);
    }

    #[test]
    fn default_std_other_channels() {
        assert_eq!(default_std(5), vec![1.0; 5]);
    }

    #[test]
    fn parse_csv_f32_valid() {
        let result = parse_csv_f32("0.485, 0.456, 0.406", "mean").unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.485).abs() < 1e-6);
    }

    #[test]
    fn parse_csv_f32_single_value() {
        let result = parse_csv_f32("0.5", "mean").unwrap();
        assert_eq!(result, vec![0.5]);
    }

    #[test]
    fn parse_csv_f32_rejects_empty_value() {
        let result = parse_csv_f32("0.5,,0.3", "mean");
        assert!(result.is_err());
    }

    #[test]
    fn parse_csv_f32_rejects_non_numeric() {
        let result = parse_csv_f32("0.5,abc,0.3", "mean");
        assert!(result.is_err());
    }

    #[test]
    fn maybe_shuffle_noop_when_false() {
        let mut order = vec![0, 1, 2, 3, 4];
        maybe_shuffle(&mut order, false);
        assert_eq!(order, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn resize_shorter_side_scales_correctly() {
        let img = RgbImage::new(200, 100);
        let resized = resize_shorter_side(&img, 50);
        assert_eq!(resized.height(), 50);
        assert_eq!(resized.width(), 100);
    }

    #[test]
    fn resize_shorter_side_landscape() {
        let img = RgbImage::new(100, 200);
        let resized = resize_shorter_side(&img, 50);
        assert_eq!(resized.width(), 50);
        assert_eq!(resized.height(), 100);
    }

    #[test]
    fn resize_shorter_side_square() {
        let img = RgbImage::new(100, 100);
        let resized = resize_shorter_side(&img, 50);
        assert_eq!(resized.width(), 50);
        assert_eq!(resized.height(), 50);
    }

    #[test]
    fn rgb_image_to_chw_layout_is_channel_first() {
        let mut img = RgbImage::new(2, 2);
        img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        img.put_pixel(1, 0, image::Rgb([0, 255, 0]));
        img.put_pixel(0, 1, image::Rgb([0, 0, 255]));
        img.put_pixel(1, 1, image::Rgb([128, 128, 128]));

        let norm = NormalizationStats {
            mean: vec![0.0, 0.0, 0.0],
            std: vec![1.0, 1.0, 1.0],
        };
        let data = rgb_image_to_chw(&img, &norm);
        assert_eq!(data.len(), 3 * 2 * 2);
        // Red channel at (0,0) should be 1.0
        assert!((data[0] - 1.0).abs() < 1e-6);
        // Green channel at (1,0) should be 1.0 (offset: 1*4 + 0*2 + 1 = 5)
        assert!((data[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn collect_image_files_finds_nested_images() {
        let root = make_temp_dir("collect-images");
        let nested = root.join("sub");
        std::fs::create_dir_all(&nested).unwrap();

        let img = GrayImage::new(1, 1);
        img.save(root.join("a.png")).unwrap();
        img.save(nested.join("b.jpg")).unwrap();
        std::fs::write(root.join("c.txt"), "not an image").unwrap();

        let files = collect_image_files(&root).unwrap();
        assert_eq!(files.len(), 2);

        std::fs::remove_dir_all(&root).unwrap();
    }
}
