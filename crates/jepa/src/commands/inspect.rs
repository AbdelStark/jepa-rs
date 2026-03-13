use std::path::Path;

use anyhow::{Context, Result};

use crate::cli::InspectArgs;
use crate::fmt_utils::truncate;

pub fn run(args: InspectArgs) -> Result<()> {
    let path = &args.path;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "safetensors" => inspect_safetensors(path),
        "onnx" => inspect_onnx(path),
        _ => {
            anyhow::bail!("Unknown file extension '.{ext}'. Expected .safetensors or .onnx");
        }
    }
}

fn inspect_safetensors(path: &Path) -> Result<()> {
    let mappings = jepa_compat::keymap::ijepa_vit_keymap();
    let checkpoint = jepa_compat::safetensors::load_checkpoint(path, &mappings)
        .context("Failed to load safetensors checkpoint")?;

    println!();
    println!("  ┌──────────────────────────────────────────────────────────────┐");
    println!("  │  Safetensors Checkpoint                                     │");
    println!("  │  Path: {:<53} │", truncate_path(path, 53));
    println!("  ├──────────────────────────────────────────────────────────────┤");
    println!("  │  Mapped tensors:   {:<40} │", checkpoint.tensors.len());
    println!(
        "  │  Unmapped keys:    {:<40} │",
        checkpoint.unmapped_keys.len()
    );
    println!("  ├──────────────────────────────────────────────────────────────┤");

    let mut total_params: usize = 0;
    let mut keys: Vec<_> = checkpoint.tensors.keys().collect();
    keys.sort();

    for key in keys.iter().take(20) {
        if let Some(t) = checkpoint.tensors.get(*key) {
            let numel: usize = t.shape.iter().product();
            total_params += numel;
            println!(
                "  │  {:<36} {:>22} │",
                truncate(key, 36),
                format_shape(&t.shape)
            );
        }
    }
    if keys.len() > 20 {
        println!(
            "  │  ... and {} more tensors                                   │",
            keys.len() - 20
        );
    }

    // Sum remaining params
    for key in keys.iter().skip(20) {
        if let Some(t) = checkpoint.tensors.get(*key) {
            let numel: usize = t.shape.iter().product();
            total_params += numel;
        }
    }

    println!("  ├──────────────────────────────────────────────────────────────┤");
    println!(
        "  │  Total parameters: {:<40} │",
        format_params(total_params)
    );
    println!("  └──────────────────────────────────────────────────────────────┘");
    println!();

    Ok(())
}

fn inspect_onnx(path: &Path) -> Result<()> {
    let info = jepa_compat::onnx::OnnxModelInfo::from_file(path)
        .context("Failed to read ONNX model info")?;

    println!();
    println!("  ┌──────────────────────────────────────────────────────────────┐");
    println!("  │  ONNX Model Info                                            │");
    println!("  │  Path: {:<53} │", truncate_path(path, 53));
    println!("  ├──────────────────────────────────────────────────────────────┤");
    println!("  │  Name:     {:<49} │", info.name);
    println!("  │  Producer: {:<49} │", info.producer);
    println!("  │  Opset:    {:<49} │", info.opset_version);
    println!("  ├──────────────────────────────────────────────────────────────┤");
    println!("  │  Inputs:                                                    │");
    for inp in &info.inputs {
        println!(
            "  │    {:<30} {:>10} {:>14} │",
            inp.name,
            format!("{:?}", inp.dtype),
            format_shape_i64(&inp.shape),
        );
    }
    println!("  │  Outputs:                                                   │");
    for out in &info.outputs {
        println!(
            "  │    {:<30} {:>10} {:>14} │",
            out.name,
            format!("{:?}", out.dtype),
            format_shape_i64(&out.shape),
        );
    }
    println!("  ├──────────────────────────────────────────────────────────────┤");
    println!("  │  Runtime Validation:                                        │");
    match jepa_compat::runtime::validate_model(path) {
        Ok(diagnostics) => {
            for diag in &diagnostics {
                println!("  │    {:<56} │", truncate(diag, 56));
            }
        }
        Err(e) => {
            println!("  │    {:<56} │", truncate(&format!("ERROR: {e}"), 56));
        }
    }
    println!("  └──────────────────────────────────────────────────────────────┘");
    println!();

    Ok(())
}

fn truncate_path(p: &Path, max: usize) -> String {
    let s = p.display().to_string();
    truncate(&s, max)
}

fn format_shape(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

fn format_shape_i64(shape: &[i64]) -> String {
    let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

fn format_params(count: usize) -> String {
    if count >= 1_000_000_000 {
        format!("{:.2}B ({count} params)", count as f64 / 1e9)
    } else if count >= 1_000_000 {
        format!("{:.1}M ({count} params)", count as f64 / 1e6)
    } else if count >= 1_000 {
        format!("{:.1}K ({count} params)", count as f64 / 1e3)
    } else {
        format!("{count} params")
    }
}
