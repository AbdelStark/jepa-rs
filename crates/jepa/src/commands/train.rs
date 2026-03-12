use anyhow::Result;
use burn::prelude::*;
use burn_ndarray::NdArray;

use jepa_core::{
    BarlowTwins, BlockMasking, CollapseRegularizer, CosineEnergy, EnergyFn, InputShape, L2Energy,
    MultiBlockMasking, SmoothL1Energy, VICReg,
};
use jepa_train::{JepaComponents, LrSchedule, WarmupCosineSchedule};
use jepa_vision::image::{TransformerPredictor, TransformerPredictorConfig};
use jepa_vision::vit::{VitConfig, VitEncoder};

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
    let (ph, pw) = vit_config.patch_size;
    let num_patches = (vit_config.image_height / ph) * (vit_config.image_width / pw);
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

    let context_encoder: VitEncoder<B> = vit_config.init(&DEVICE);
    let target_encoder: VitEncoder<B> = vit_config.init(&DEVICE);
    let predictor: TransformerPredictor<B> = predictor_config.init(&DEVICE);

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
                &context_encoder,
                &target_encoder,
                &predictor,
                &L2Energy,
                &VICReg::default(),
                &args.masking,
                &block_masking,
                &multi_block_masking,
                image_h,
                image_w,
            );
        }
        (EnergyChoice::L2, RegularizerChoice::BarlowTwins) => {
            run_loop(
                &args,
                &context_encoder,
                &target_encoder,
                &predictor,
                &L2Energy,
                &BarlowTwins::default(),
                &args.masking,
                &block_masking,
                &multi_block_masking,
                image_h,
                image_w,
            );
        }
        (EnergyChoice::Cosine, RegularizerChoice::Vicreg) => {
            run_loop(
                &args,
                &context_encoder,
                &target_encoder,
                &predictor,
                &CosineEnergy,
                &VICReg::default(),
                &args.masking,
                &block_masking,
                &multi_block_masking,
                image_h,
                image_w,
            );
        }
        (EnergyChoice::Cosine, RegularizerChoice::BarlowTwins) => {
            run_loop(
                &args,
                &context_encoder,
                &target_encoder,
                &predictor,
                &CosineEnergy,
                &BarlowTwins::default(),
                &args.masking,
                &block_masking,
                &multi_block_masking,
                image_h,
                image_w,
            );
        }
        (EnergyChoice::SmoothL1, RegularizerChoice::Vicreg) => {
            run_loop(
                &args,
                &context_encoder,
                &target_encoder,
                &predictor,
                &SmoothL1Energy::new(1.0),
                &VICReg::default(),
                &args.masking,
                &block_masking,
                &multi_block_masking,
                image_h,
                image_w,
            );
        }
        (EnergyChoice::SmoothL1, RegularizerChoice::BarlowTwins) => {
            run_loop(
                &args,
                &context_encoder,
                &target_encoder,
                &predictor,
                &SmoothL1Energy::new(1.0),
                &BarlowTwins::default(),
                &args.masking,
                &block_masking,
                &multi_block_masking,
                image_h,
                image_w,
            );
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_loop<EF, CR>(
    args: &TrainArgs,
    context_encoder: &VitEncoder<B>,
    target_encoder: &VitEncoder<B>,
    predictor: &TransformerPredictor<B>,
    energy_fn: &EF,
    regularizer: &CR,
    masking_choice: &MaskingChoice,
    block_masking: &BlockMasking,
    multi_block_masking: &MultiBlockMasking,
    image_h: usize,
    image_w: usize,
) where
    EF: EnergyFn<B>,
    CR: CollapseRegularizer<B>,
{
    let shape = InputShape::Image {
        height: image_h,
        width: image_w,
    };
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

        // Use the appropriate masking strategy
        match masking_choice {
            MaskingChoice::Block => {
                let components = JepaComponents::new(
                    context_encoder,
                    target_encoder,
                    predictor,
                    energy_fn,
                    regularizer,
                    block_masking,
                    args.reg_weight,
                );
                let output = components.forward_step(&input, &shape, &mut rng);
                print_step(step, &output, lr, args);
            }
            MaskingChoice::MultiBlock => {
                let components = JepaComponents::new(
                    context_encoder,
                    target_encoder,
                    predictor,
                    energy_fn,
                    regularizer,
                    multi_block_masking,
                    args.reg_weight,
                );
                let output = components.forward_step(&input, &shape, &mut rng);
                print_step(step, &output, lr, args);
            }
        }
    }

    println!("  └────────┴──────────────┴──────────────┴──────────────┴──────────┘");
    println!();
    println!("  Training complete.");
    println!();
}

fn print_step(step: usize, output: &jepa_train::JepaForwardOutput<B>, lr: f64, args: &TrainArgs) {
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
