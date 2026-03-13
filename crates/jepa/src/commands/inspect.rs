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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    #[test]
    fn truncate_path_short() {
        let p = Path::new("/tmp/foo.onnx");
        let result = truncate_path(p, 50);
        assert_eq!(result, "/tmp/foo.onnx");
    }

    #[test]
    fn truncate_path_long() {
        let p = Path::new(
            "/very/long/path/that/exceeds/the/maximum/allowed/characters/model.safetensors",
        );
        let result = truncate_path(p, 20);
        assert!(result.ends_with("..."));
        assert_eq!(result.len(), 20);
    }

    #[test]
    fn format_shape_usize_basic() {
        assert_eq!(format_shape(&[3, 224, 224]), "[3, 224, 224]");
        assert_eq!(format_shape(&[768]), "[768]");
        assert_eq!(format_shape(&[]), "[]");
    }

    #[test]
    fn format_shape_i64_basic() {
        assert_eq!(format_shape_i64(&[1, 3, 224, 224]), "[1, 3, 224, 224]");
        assert_eq!(format_shape_i64(&[-1, 768]), "[-1, 768]");
    }

    #[test]
    fn format_params_billions() {
        let result = format_params(1_500_000_000);
        assert!(result.starts_with("1.50B"));
        assert!(result.contains("1500000000 params"));
    }

    #[test]
    fn format_params_millions() {
        let result = format_params(86_000_000);
        assert!(result.starts_with("86.0M"));
    }

    #[test]
    fn format_params_thousands() {
        let result = format_params(50_000);
        assert!(result.starts_with("50.0K"));
    }

    #[test]
    fn format_params_small() {
        assert_eq!(format_params(42), "42 params");
    }

    #[test]
    fn run_rejects_unknown_extension() {
        let args = InspectArgs {
            path: PathBuf::from("/tmp/model.bin"),
        };
        let result = run(args);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Unknown file extension"));
    }
}
