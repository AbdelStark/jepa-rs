use anyhow::{Context, Result};
use burn::prelude::*;
use burn_ndarray::NdArray;

use jepa_core::{
    BarlowTwins, BlockMasking, CollapseRegularizer, CosineEnergy, EnergyFn, InputShape, L2Energy,
    MaskingStrategy, MultiBlockMasking, SmoothL1Energy, VICReg,
};
use jepa_train::{LrSchedule, WarmupCosineSchedule};
use jepa_vision::image::{
    IJepa, IJepaConfig, StrictIJepaForwardOutput, TransformerPredictorConfig,
};
use jepa_vision::vit::VitConfig;

use crate::cli::{ArchPreset, EnergyChoice, MaskingChoice, RegularizerChoice, TrainArgs};

type B = NdArray<f32>;
const DEVICE: burn_ndarray::NdArrayDevice = burn_ndarray::NdArrayDevice::Cpu;

pub fn run(args: TrainArgs) -> Result<()> {
    println!();
    print_banner();

    let vit_config = match args.preset {
        ArchPreset::VitBase16 => VitConfig::vit_base_patch16(),
        ArchPreset::VitSmall16 => VitConfig::vit_small_patch16(),
        ArchPreset::VitLarge16 => VitConfig::vit_large_patch16(),
        ArchPreset::VitHuge14 => VitConfig::vit_huge_patch14(),
    };

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
    println!();

    let predictor_config = TransformerPredictorConfig {
        encoder_embed_dim: embed_dim,
        predictor_embed_dim: embed_dim / 4,
        num_layers: 6,
        num_heads: vit_config.num_heads,
        max_target_len: num_patches,
    };

    let model: IJepa<B> = IJepaConfig {
        encoder: vit_config.clone(),
        predictor: predictor_config,
    }
    .init(&DEVICE);

    let _lr_schedule = WarmupCosineSchedule::new(args.lr, args.warmup, args.steps);
    let mut _ema = jepa_core::Ema::with_cosine_schedule(args.ema_momentum, args.steps);

    let block_masking = BlockMasking {
        num_targets: 4,
        target_scale: (0.15, 0.4),
        target_aspect_ratio: (0.75, 1.5),
    };
    let multi_block_masking = MultiBlockMasking {
        mask_ratio: 0.4,
        num_blocks: 4,
    };

    // Dispatch on energy/regularizer/masking combinations
    match (&args.energy, &args.regularizer) {
        (EnergyChoice::L2, RegularizerChoice::Vicreg) => {
            run_loop(
                &args,
                &model,
                &L2Energy,
                &VICReg::default(),
                &args.masking,
                &block_masking,
                &multi_block_masking,
                &mask_shape,
                image_h,
                image_w,
            )?;
        }
        (EnergyChoice::L2, RegularizerChoice::BarlowTwins) => {
            run_loop(
                &args,
                &model,
                &L2Energy,
                &BarlowTwins::default(),
                &args.masking,
                &block_masking,
                &multi_block_masking,
                &mask_shape,
                image_h,
                image_w,
            )?;
        }
        (EnergyChoice::Cosine, RegularizerChoice::Vicreg) => {
            run_loop(
                &args,
                &model,
                &CosineEnergy,
                &VICReg::default(),
                &args.masking,
                &block_masking,
                &multi_block_masking,
                &mask_shape,
                image_h,
                image_w,
            )?;
        }
        (EnergyChoice::Cosine, RegularizerChoice::BarlowTwins) => {
            run_loop(
                &args,
                &model,
                &CosineEnergy,
                &BarlowTwins::default(),
                &args.masking,
                &block_masking,
                &multi_block_masking,
                &mask_shape,
                image_h,
                image_w,
            )?;
        }
        (EnergyChoice::SmoothL1, RegularizerChoice::Vicreg) => {
            run_loop(
                &args,
                &model,
                &SmoothL1Energy::new(1.0),
                &VICReg::default(),
                &args.masking,
                &block_masking,
                &multi_block_masking,
                &mask_shape,
                image_h,
                image_w,
            )?;
        }
        (EnergyChoice::SmoothL1, RegularizerChoice::BarlowTwins) => {
            run_loop(
                &args,
                &model,
                &SmoothL1Energy::new(1.0),
                &BarlowTwins::default(),
                &args.masking,
                &block_masking,
                &multi_block_masking,
                &mask_shape,
                image_h,
                image_w,
            )?;
        }
    }

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
fn run_loop<EF, CR>(
    args: &TrainArgs,
    model: &IJepa<B>,
    energy_fn: &EF,
    regularizer: &CR,
    masking_choice: &MaskingChoice,
    block_masking: &BlockMasking,
    multi_block_masking: &MultiBlockMasking,
    input_shape: &InputShape,
    image_h: usize,
    image_w: usize,
) -> Result<()>
where
    EF: EnergyFn<B>,
    CR: CollapseRegularizer<B>,
{
    let mut rng = rand::rng();

    let lr_schedule = WarmupCosineSchedule::new(args.lr, args.warmup, args.steps);

    println!("  ┌────────┬──────────────┬──────────────┬──────────────┬──────────┐");
    println!("  │  Step  │    Energy    │     Reg      │  Total Loss  │    LR    │");
    println!("  ├────────┼──────────────┼──────────────┼──────────────┼──────────┤");

    for step in 0..args.steps {
        let lr = lr_schedule.get_lr(step);

        let input: Tensor<B, 4> = Tensor::random(
            [args.batch_size, 3, image_h, image_w],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &DEVICE,
        );

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
    }

    println!("  └────────┴──────────────┴──────────────┴──────────────┴──────────┘");
    println!();
    println!("  Training complete.");
    println!();

    Ok(())
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

fn print_banner() {
    println!("  ╔══════════════════════════════════════════════════════════════╗");
    println!("  ║            JEPA Training — Synthetic Demo                  ║");
    println!("  ║                                                            ║");
    println!("  ║  Note: Using synthetic random data for demonstration.      ║");
    println!("  ║  Real training requires a dataset loader.                  ║");
    println!("  ╚══════════════════════════════════════════════════════════════╝");
    println!();
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
        let model: IJepa<B> = IJepaConfig {
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
}
