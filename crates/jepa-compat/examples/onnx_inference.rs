//! End-to-end ONNX inference example for JEPA models.
//!
//! Demonstrates loading a pre-exported ONNX model and running inference
//! to extract image representations using the tract inference engine.
//!
//! ## Prerequisites
//!
//! Export an I-JEPA model to ONNX first:
//! ```bash
//! # Option 1: Create a tiny test model (no GPU or pretrained weights needed)
//! python scripts/export_ijepa_onnx.py --tiny-test
//!
//! # Option 2: Export a real pretrained model
//! python scripts/export_ijepa_onnx.py --model vit_h14
//! ```
//!
//! ## Run
//!
//! ```bash
//! cargo run -p jepa-compat --example onnx_inference -- ijepa_tiny_test.onnx
//! ```

use jepa_compat::registry;
use jepa_compat::runtime::{self, OnnxSession};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: onnx_inference <model.onnx> [--info-only]");
        eprintln!();
        eprintln!("Export a model first:");
        eprintln!("  python scripts/export_ijepa_onnx.py --tiny-test");
        eprintln!("  cargo run -p jepa-compat --example onnx_inference -- ijepa_tiny_test.onnx");
        eprintln!();
        eprintln!("Available pretrained models:");
        print!("{}", registry::format_model_table());
        std::process::exit(1);
    }

    let model_path = &args[1];
    let info_only = args.get(2).is_some_and(|a| a == "--info-only");

    println!("=== JEPA ONNX Inference ===\n");

    // Step 1: Inspect the model
    println!("Loading model: {}", model_path);
    let info = runtime::inspect_model(model_path).unwrap_or_else(|e| {
        eprintln!("Failed to inspect model: {e}");
        std::process::exit(1);
    });
    print!("{}", runtime::format_model_summary(&info));
    println!();

    if info_only {
        return;
    }

    // Step 2: Create inference session
    println!("Creating inference session...");
    let session = OnnxSession::from_path(model_path).unwrap_or_else(|e| {
        // If the model has dynamic shapes, try with a fixed input shape
        eprintln!("Direct load failed ({e}), trying with fixed input shape...");
        let input = &info.inputs[0];
        let shape: Vec<usize> = input
            .shape
            .iter()
            .map(|&d| if d < 0 { 1 } else { d as usize })
            .collect();
        OnnxSession::from_path_with_input_shape(model_path, &shape).unwrap_or_else(|e2| {
            eprintln!("Failed to create session: {e2}");
            std::process::exit(1);
        })
    });
    println!("Session created successfully\n");

    // Step 3: Prepare input
    let input_info = &info.inputs[0];
    let input_shape: Vec<usize> = input_info
        .shape
        .iter()
        .map(|&d| if d < 0 { 1 } else { d as usize })
        .collect();
    let input_size: usize = input_shape.iter().product();

    println!("Input shape: {:?}", input_shape);
    println!("Input size: {} elements", input_size);

    // Create a synthetic input (in practice, this would be a real image)
    // Using ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    let input_data: Vec<f32> = (0..input_size)
        .map(|i| {
            let channel = (i / (input_shape[2] * input_shape[3])) % input_shape[1];
            let means = [0.485f32, 0.456, 0.406];
            let stds = [0.229f32, 0.224, 0.225];
            // Normalized zero image
            -means[channel % 3] / stds[channel % 3]
        })
        .collect();
    println!("Created synthetic input (normalized zero image)\n");

    // Step 4: Run inference
    println!("Running inference...");
    let start = std::time::Instant::now();
    let output = session
        .run_f32(&input_shape, &input_data)
        .unwrap_or_else(|e| {
            eprintln!("Inference failed: {e}");
            std::process::exit(1);
        });
    let elapsed = start.elapsed();

    println!("Output shape: {:?}", output.shape);
    println!("Output size: {} elements", output.len());
    println!("Inference time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!();

    // Step 5: Analyze output
    if let Some((data, tokens, embed_dim)) = output.as_token_embeddings() {
        println!("Representations:");
        println!("  Tokens: {}", tokens);
        println!("  Embedding dim: {}", embed_dim);

        // Compute basic stats
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let variance: f32 =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!("  Mean: {:.6}", mean);
        println!("  Std:  {:.6}", variance.sqrt());
        println!("  Min:  {:.6}", min);
        println!("  Max:  {:.6}", max);

        // Show first few token embeddings
        println!("\n  First 3 token embeddings (first 5 dims):");
        for t in 0..tokens.min(3) {
            let start = t * embed_dim;
            let dims: Vec<String> = data[start..start + embed_dim.min(5)]
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect();
            println!("    Token {}: [{}...]", t, dims.join(", "));
        }
    }

    println!("\n=== Done ===");
}
