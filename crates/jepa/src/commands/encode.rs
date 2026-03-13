use anyhow::Result;
use burn::prelude::*;
use burn_ndarray::NdArray;

use jepa_core::Encoder;
use jepa_vision::vit::VitConfig;

use crate::cli::{ArchPreset, EncodeArgs};

type B = NdArray<f32>;
const DEVICE: burn_ndarray::NdArrayDevice = burn_ndarray::NdArrayDevice::Cpu;

pub fn run(args: EncodeArgs) -> Result<()> {
    let vit_config = match args.preset {
        ArchPreset::VitBase16 => VitConfig::vit_base_patch16(),
        ArchPreset::VitSmall16 => VitConfig::vit_small_patch16(),
        ArchPreset::VitLarge16 => VitConfig::vit_large_patch16(),
        ArchPreset::VitHuge14 => VitConfig::vit_huge_patch14(),
    };

    let embed_dim = vit_config.embed_dim;
    let (ph, pw) = vit_config.patch_size;
    let num_patches = (args.height / ph) * (args.width / pw);

    println!();
    println!("  ╔══════════════════════════════════════════════════════════════╗");
    println!("  ║                      JEPA Encoder                          ║");
    println!("  ╠══════════════════════════════════════════════════════════════╣");
    println!("  ║  Architecture:  {:<43} ║", format!("{:?}", args.preset));
    println!("  ║  Embed dim:     {:<43} ║", embed_dim);
    println!(
        "  ║  Input size:    {:<43} ║",
        format!("{}x{}", args.height, args.width)
    );
    println!("  ║  Num patches:   {:<43} ║", num_patches);
    println!(
        "  ║  Checkpoint:    {:<43} ║",
        truncate(&args.model.display().to_string(), 43)
    );
    println!("  ╚══════════════════════════════════════════════════════════════╝");
    println!();

    let encoder = vit_config.init::<B>(&DEVICE);

    println!("  Note: Using randomly initialized weights (checkpoint loading");
    println!("  into burn modules requires weight injection support).");
    println!();

    for i in 0..args.num_samples {
        let input: Tensor<B, 4> = Tensor::random(
            [1, 3, args.height, args.width],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &DEVICE,
        );

        let repr = encoder.encode(&input);
        let shape = repr.embeddings.dims();

        println!("  Sample {}/{}", i + 1, args.num_samples);
        println!(
            "    Output shape: [{}, {}, {}]",
            shape[0], shape[1], shape[2]
        );

        let flat = repr.embeddings.reshape([shape[0] * shape[1], shape[2]]);
        let show_dims = embed_dim.min(8);
        let first_token = flat.slice([0..1, 0..show_dims]);
        let vals: Vec<f32> = first_token
            .to_data()
            .to_vec()
            .map_err(|e| anyhow::anyhow!("failed to read tensor data: {e}"))?;
        print!("    First token (first {} dims): [", vals.len());
        for (j, v) in vals.iter().enumerate() {
            if j > 0 {
                print!(", ");
            }
            print!("{v:.4}");
        }
        println!("]");

        if repr.mask.is_some() {
            println!("    Mask: present");
        } else {
            println!("    Mask: none");
        }
        println!();
    }

    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max - 3])
    }
}
