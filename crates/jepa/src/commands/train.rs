use anyhow::{bail, Context, Result};
use burn::backend::Autodiff;
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn_ndarray::NdArray;
use rand::seq::SliceRandom;

use jepa_core::{
    BarlowTwins, BlockMasking, CollapseRegularizer, CosineEnergy, Ema, EnergyFn, InputShape,
    L2Energy, MaskingStrategy, MultiBlockMasking, SmoothL1Energy, VICReg,
};
use jepa_train::{LrSchedule, WarmupCosineSchedule};
use jepa_vision::image::{
    IJepa, IJepaConfig, StrictIJepaForwardOutput, TransformerPredictorConfig,
};
use jepa_vision::vit::VitConfig;

use crate::cli::{ArchPreset, EnergyChoice, MaskingChoice, RegularizerChoice, TrainArgs};

type B = Autodiff<NdArray<f32>>;
const DEVICE: burn_ndarray::NdArrayDevice = burn_ndarray::NdArrayDevice::Cpu;

pub fn run(args: TrainArgs) -> Result<()> {
    println!();

    let vit_config = match args.preset {
        ArchPreset::VitBase16 => VitConfig::vit_base_patch16(),
        ArchPreset::VitSmall16 => VitConfig::vit_small_patch16(),
        ArchPreset::VitLarge16 => VitConfig::vit_large_patch16(),
        ArchPreset::VitHuge14 => VitConfig::vit_huge_patch14(),
    };

    let mut batch_source = BatchSource::from_args(&args, &vit_config)?;
    print_banner(&batch_source);

    let embed_dim = vit_config.embed_dim;
    let mask_shape = training_input_shape(&vit_config);
    let (ph, pw) = vit_config.patch_size;
    let num_patches = mask_shape.total_tokens();
    let image_h = vit_config.image_height;
    let image_w = vit_config.image_width;

    println!("  Architecture:   {:?}", args.preset);
    println!("  Embed dim:      {embed_dim}");
    println!("  Patch size:     {ph}x{pw}");
    println!("  Image size:     {image_h}x{image_w}");
    println!("  Num patches:    {num_patches}");
    println!("  Steps:          {}", args.steps);
    println!("  Warmup:         {}", args.warmup);
    println!("  LR:             {}", args.lr);
    println!("  Batch size:     {}", args.batch_size);
    println!("  Masking:        {:?}", args.masking);
    println!("  Energy:         {:?}", args.energy);
    println!("  Regularizer:    {:?}", args.regularizer);
    println!("  Reg weight:     {}", args.reg_weight);
    println!("  EMA momentum:   {}", args.ema_momentum);
    println!("  Data source:    {}", batch_source.describe());
    println!();

    let predictor_config = TransformerPredictorConfig {
        encoder_embed_dim: embed_dim,
        predictor_embed_dim: embed_dim / 4,
        num_layers: 6,
        num_heads: vit_config.num_heads,
        max_target_len: num_patches,
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
        )?,
    };

    // Keep the final model alive until the end of the command.
    let _ = model;

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
) -> Result<IJepa<B>>
where
    EF: EnergyFn<B>,
    CR: CollapseRegularizer<B>,
    O: Optimizer<IJepa<B>, B>,
{
    let mut rng = rand::rng();
    let lr_schedule = WarmupCosineSchedule::new(args.lr, args.warmup, args.steps);

    println!("  ┌────────┬──────────────┬──────────────┬──────────────┬──────────┐");
    println!("  │  Step  │    Energy    │     Reg      │  Total Loss  │    LR    │");
    println!("  ├────────┼──────────────┼──────────────┼──────────────┼──────────┤");

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

        print_step(step, &output, lr, args);

        let grads = GradientsParams::from_grads(output.total_loss.backward(), &model);
        model = optimizer.step(lr, model, grads);
        model.target_encoder = model
            .target_encoder
            .clone()
            .ema_update_from(&model.context_encoder, ema, step)
            .no_grad();
    }

    println!("  └────────┴──────────────┴──────────────┴──────────────┴──────────┘");
    println!();
    println!("  Training complete.");
    println!();

    Ok(model)
}

fn print_step(step: usize, output: &StrictIJepaForwardOutput<B>, lr: f64, args: &TrainArgs) {
    let energy_val: f32 = output.energy.value.clone().into_scalar();
    let reg_val: f32 = output.regularization.clone().into_scalar();
    let total_val: f32 = output.total_loss.clone().into_scalar();

    if step % args.log_interval == 0 || step == args.steps - 1 {
        println!(
            "  │ {:>5}  │ {:>12.6} │ {:>12.6} │ {:>12.6} │ {:>8.2e} │",
            step, energy_val, reg_val, total_val, lr,
        );
    }
}

fn print_banner(batch_source: &BatchSource) {
    println!("  ╔══════════════════════════════════════════════════════════════╗");
    match batch_source {
        BatchSource::Synthetic { .. } => {
            println!("  ║            JEPA Training — Synthetic Demo                  ║");
            println!("  ║                                                            ║");
            println!("  ║  Optimizer and EMA are active on synthetic random data.    ║");
            println!("  ║  Pass --dataset to train on real stored image tensors.     ║");
        }
        BatchSource::Dataset(dataset) => {
            println!("  ║            JEPA Training — Dataset Mode                    ║");
            println!("  ║                                                            ║");
            println!("  ║  Strict masking, AdamW, and EMA are active.                ║");
            println!(
                "  ║  Loaded {:>5} image tensor(s) from safetensors dataset.     ║",
                dataset.num_samples
            );
        }
    }
    println!("  ╚══════════════════════════════════════════════════════════════╝");
    println!();
}

enum BatchSource {
    Synthetic {
        channels: usize,
        height: usize,
        width: usize,
    },
    Dataset(ImageTensorDataset),
}

impl BatchSource {
    fn from_args(args: &TrainArgs, vit_config: &VitConfig) -> Result<Self> {
        match &args.dataset {
            Some(path) => Ok(Self::Dataset(ImageTensorDataset::from_safetensors(
                path,
                &args.dataset_key,
                vit_config.in_channels,
                vit_config.image_height,
                vit_config.image_width,
            )?)),
            None => Ok(Self::Synthetic {
                channels: vit_config.in_channels,
                height: vit_config.image_height,
                width: vit_config.image_width,
            }),
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Synthetic { .. } => "synthetic random tensors".to_string(),
            Self::Dataset(dataset) => format!(
                "{}:{} [{} samples]",
                dataset.path.display(),
                dataset.tensor_key,
                dataset.num_samples
            ),
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
            Self::Dataset(dataset) => dataset.next_batch(batch_size, device),
        }
    }
}

struct ImageTensorDataset {
    path: std::path::PathBuf,
    tensor_key: String,
    data: Vec<f32>,
    num_samples: usize,
    channels: usize,
    height: usize,
    width: usize,
    order: Vec<usize>,
    cursor: usize,
}

impl ImageTensorDataset {
    fn from_safetensors(
        path: &std::path::Path,
        tensor_key: &str,
        expected_channels: usize,
        expected_height: usize,
        expected_width: usize,
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
            expected_channels,
            expected_height,
            expected_width,
        )
    }

    fn from_loaded_tensor(
        path: std::path::PathBuf,
        tensor_key: String,
        tensor: TensorData,
        expected_channels: usize,
        expected_height: usize,
        expected_width: usize,
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

        if *channels != expected_channels || *height != expected_height || *width != expected_width
        {
            bail!(
                "dataset tensor `{tensor_key}` in {} must match [{} , {}, {}, {}], got {:?}",
                path.display(),
                "N",
                expected_channels,
                expected_height,
                expected_width,
                tensor.shape
            );
        }

        let data = tensor.to_vec::<f32>().map_err(|err| {
            anyhow::anyhow!("failed to decode dataset tensor `{tensor_key}`: {err}")
        })?;
        let mut order: Vec<usize> = (0..*num_samples).collect();
        order.shuffle(&mut rand::rng());

        Ok(Self {
            path,
            tensor_key,
            data,
            num_samples: *num_samples,
            channels: *channels,
            height: *height,
            width: *width,
            order,
            cursor: 0,
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
                self.order.shuffle(&mut rand::rng());
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

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
            std::path::PathBuf::from("test.safetensors"),
            "images".to_string(),
            TensorData::new(vec![0.5f32; 3 * 8 * 8], [3, 1, 8, 8]),
            1,
            8,
            8,
        )
        .expect("test dataset should load");

        let mut batch_source = BatchSource::Dataset(dataset);
        let batch = batch_source
            .next_batch(2, &DEVICE)
            .expect("dataset batch should materialize");
        assert_eq!(batch.dims(), [2, 1, 8, 8]);
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
}
