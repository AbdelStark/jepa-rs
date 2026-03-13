use anyhow::{Context, Result};

use crate::cli::{CheckpointArgs, KeymapPreset};
use crate::fmt_utils::{format_params, truncate};

pub fn run(args: CheckpointArgs) -> Result<()> {
    let mappings = match args.keymap {
        KeymapPreset::Ijepa => jepa_compat::keymap::ijepa_vit_keymap(),
        KeymapPreset::Vjepa => jepa_compat::keymap::vjepa_vit_keymap(),
        KeymapPreset::None => vec![],
    };

    let checkpoint = jepa_compat::safetensors::load_checkpoint(&args.path, &mappings)
        .context("Failed to load checkpoint")?;

    println!();
    println!("  ╔══════════════════════════════════════════════════════════════╗");
    println!("  ║  Checkpoint Analysis                                       ║");
    println!("  ╠══════════════════════════════════════════════════════════════╣");
    println!(
        "  ║  File:       {:<47} ║",
        truncate(&args.path.display().to_string(), 47)
    );
    println!("  ║  Keymap:     {:<47} ║", format!("{:?}", args.keymap));
    println!("  ║  Tensors:    {:<47} ║", checkpoint.tensors.len());
    println!("  ║  Unmapped:   {:<47} ║", checkpoint.unmapped_keys.len());
    println!("  ╠══════════════════════════════════════════════════════════════╣");

    let mut total_params: usize = 0;
    let mut total_bytes: usize = 0;
    let mut keys: Vec<_> = checkpoint.tensors.keys().collect();
    keys.sort();

    if args.verbose {
        println!("  ║ {:<34} {:<12} {:>10} ║", "Tensor", "Shape", "Params");
        println!("  ╠══════════════════════════════════════════════════════════════╣");

        for key in &keys {
            if let Some(t) = checkpoint.tensors.get(*key) {
                let numel: usize = t.shape.iter().product();
                total_params += numel;
                total_bytes += numel * 4; // f32

                let shape_str = format_shape(&t.shape);
                println!(
                    "  ║ {:<34} {:<12} {:>10} ║",
                    truncate(key, 34),
                    shape_str,
                    format_params(numel),
                );
            }
        }
    } else {
        // Summary by layer groups
        let mut groups: std::collections::BTreeMap<String, (usize, usize)> =
            std::collections::BTreeMap::new();

        for key in &keys {
            let Some(t) = checkpoint.tensors.get(*key) else {
                continue;
            };
            let numel: usize = t.shape.iter().product();
            total_params += numel;
            total_bytes += numel * 4;

            let group = key.split('.').take(2).collect::<Vec<_>>().join(".");
            let entry = groups.entry(group).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += numel;
        }

        println!(
            "  ║ {:<34} {:<10} {:>12} ║",
            "Layer Group", "Tensors", "Parameters"
        );
        println!("  ╠══════════════════════════════════════════════════════════════╣");

        for (group, (count, params)) in &groups {
            println!(
                "  ║ {:<34} {:<10} {:>12} ║",
                truncate(group, 34),
                count,
                format_params(*params),
            );
        }
    }

    println!("  ╠══════════════════════════════════════════════════════════════╣");
    println!(
        "  ║  Total parameters:  {:<39} ║",
        format_params(total_params)
    );
    println!(
        "  ║  Estimated size:    {:<39} ║",
        format_bytes(total_bytes)
    );
    println!("  ╚══════════════════════════════════════════════════════════════╝");

    if !checkpoint.unmapped_keys.is_empty() {
        println!();
        println!("  Unmapped keys ({}):", checkpoint.unmapped_keys.len());
        for key in checkpoint.unmapped_keys.iter().take(10) {
            println!("    - {key}");
        }
        if checkpoint.unmapped_keys.len() > 10 {
            println!("    ... and {} more", checkpoint.unmapped_keys.len() - 10);
        }
    }

    println!();
    Ok(())
}

fn format_shape(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("[{}]", parts.join(","))
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_shape_empty() {
        assert_eq!(format_shape(&[]), "[]");
    }

    #[test]
    fn format_shape_single_dim() {
        assert_eq!(format_shape(&[768]), "[768]");
    }

    #[test]
    fn format_shape_multi_dim() {
        assert_eq!(format_shape(&[3, 224, 224]), "[3,224,224]");
    }

    #[test]
    fn format_bytes_small() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1023), "1023 B");
    }

    #[test]
    fn format_bytes_kilobytes() {
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
    }

    #[test]
    fn format_bytes_megabytes() {
        assert_eq!(format_bytes(1_048_576), "1.0 MB");
        assert_eq!(format_bytes(5_242_880), "5.0 MB");
    }

    #[test]
    fn format_bytes_gigabytes() {
        assert_eq!(format_bytes(1_073_741_824), "1.00 GB");
        assert_eq!(format_bytes(2_684_354_560), "2.50 GB");
    }
}
