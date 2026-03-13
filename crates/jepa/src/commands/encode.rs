use anyhow::{Context, Result};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn_ndarray::NdArray;
use image::imageops::{resize, FilterType};
use image::RgbImage;

use jepa_compat::keymap::ijepa_vit_keymap;
use jepa_compat::safetensors::Checkpoint;
use jepa_core::Encoder;
use jepa_vision::vit::VitConfig;

use crate::cli::{ArchPreset, EncodeArgs};
use crate::demo::{self, InferenceDemoId};
use crate::fmt_utils::truncate;

type B = NdArray<f32>;
const DEVICE: burn_ndarray::NdArrayDevice = burn_ndarray::NdArrayDevice::Cpu;
const DEFAULT_RGB_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DEFAULT_RGB_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone)]
pub(crate) struct InferenceDemoRunSummary {
    pub preset: ArchPreset,
    pub embed_dim: usize,
    pub patch_size: (usize, usize),
    pub image_size: (usize, usize),
    pub num_patches: usize,
    pub num_samples: usize,
    pub input_description: String,
    pub model_description: String,
}

#[derive(Debug, Clone)]
pub(crate) struct InferencePhaseUpdate {
    pub title: String,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub(crate) struct InferenceSampleMetrics {
    pub sample_index: usize,
    pub total_samples: usize,
    pub sample_label: String,
    pub output_shape: [usize; 3],
    pub inference_time_ms: f64,
    pub first_token_preview: Vec<f32>,
    pub embedding_mean: f64,
    pub embedding_std: f64,
    pub mean_token_l2_norm: f64,
    pub mask_present: bool,
}

pub(crate) trait InferenceDemoReporter {
    fn on_run_started(&mut self, _summary: &InferenceDemoRunSummary) {}

    fn on_phase(&mut self, _phase: &InferencePhaseUpdate) {}

    fn on_sample(&mut self, _metrics: &InferenceSampleMetrics) {}

    fn on_run_complete(&mut self, _summary: &InferenceDemoRunSummary) {}
}

pub fn run(args: EncodeArgs) -> Result<()> {
    let ext = args
        .model
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "onnx" => run_onnx(args),
        "safetensors" => run_safetensors(args),
        _ => run_demo(args),
    }
}

pub(crate) fn run_inference_demo_with_reporter<R>(
    demo_id: InferenceDemoId,
    reporter: &mut R,
) -> Result<()>
where
    R: InferenceDemoReporter,
{
    let preset = demo_id.preset();
    let vit_config = vit_config_for_preset(&preset);
    let (height, width) = demo_id.input_size();
    let (patch_h, patch_w) = vit_config.patch_size;
    let num_patches = (height / patch_h) * (width / patch_w);
    let sample_count = demo_id.sample_count();

    let summary = InferenceDemoRunSummary {
        preset: preset.clone(),
        embed_dim: vit_config.embed_dim,
        patch_size: vit_config.patch_size,
        image_size: (height, width),
        num_patches,
        num_samples: sample_count,
        input_description: format!(
            "{} deterministic image pattern(s) shared with the demo dataset",
            sample_count
        ),
        model_description: demo_id.engine_note().to_string(),
    };
    reporter.on_run_started(&summary);

    reporter.on_phase(&InferencePhaseUpdate {
        title: "Encoder initialized".to_string(),
        detail: format!(
            "{:?} with {} patches at {}x{} and embedding dim {}.",
            summary.preset, summary.num_patches, height, width, summary.embed_dim
        ),
    });

    let encoder = vit_config.init::<B>(&DEVICE);

    reporter.on_phase(&InferencePhaseUpdate {
        title: "Preparing deterministic inputs".to_string(),
        detail: format!(
            "Resizing pattern images to {}x{} and applying ImageNet normalization.",
            height, width
        ),
    });

    let demo_inputs = build_demo_inputs(height, width, sample_count)?;
    for (index, sample) in demo_inputs.iter().enumerate() {
        reporter.on_phase(&InferencePhaseUpdate {
            title: "Running encoder inference".to_string(),
            detail: format!(
                "Sample {}/{} `{}` is being patchified and encoded.",
                index + 1,
                demo_inputs.len(),
                sample.label
            ),
        });

        let start = std::time::Instant::now();
        let repr = encoder.encode(&sample.tensor);
        let elapsed = start.elapsed();
        let metrics = summarize_inference_sample(
            repr,
            &sample.label,
            index,
            demo_inputs.len(),
            elapsed.as_secs_f64() * 1000.0,
        )?;
        reporter.on_sample(&metrics);
    }

    reporter.on_run_complete(&summary);

    Ok(())
}

fn run_onnx(args: EncodeArgs) -> Result<()> {
    use jepa_compat::runtime::OnnxEncoder;

    println!();
    println!("  ╔══════════════════════════════════════════════════════════════╗");
    println!("  ║                    JEPA ONNX Encoder                        ║");
    println!("  ╠══════════════════════════════════════════════════════════════╣");
    println!(
        "  ║  Model:         {:<43} ║",
        truncate(&args.model.display().to_string(), 43)
    );
    println!(
        "  ║  Input size:    {:<43} ║",
        format!("{}x{}", args.height, args.width)
    );

    let input_shape = [1usize, 3, args.height, args.width];

    // Try direct loading first; fall back to fixed input shape for dynamic dims.
    let encoder = match OnnxEncoder::from_path(&args.model) {
        Ok(enc) => enc,
        Err(_) => OnnxEncoder::from_path_with_input_shape(&args.model, &input_shape)
            .map_err(|e| anyhow::anyhow!("Failed to load ONNX model: {e}"))?,
    };

    let info = encoder.info();
    println!("  ║  Graph name:    {:<43} ║", info.name);
    println!("  ║  Producer:      {:<43} ║", info.producer);
    println!("  ║  Opset:         {:<43} ║", info.opset_version);
    println!("  ║  Embed dim:     {:<43} ║", encoder.embed_dim());
    println!("  ╚══════════════════════════════════════════════════════════════╝");
    println!();

    for i in 0..args.num_samples {
        let input: Tensor<B, 4> = Tensor::random(
            [1, 3, args.height, args.width],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &DEVICE,
        );

        let start = std::time::Instant::now();
        let repr = encoder.encode(&input);
        let elapsed = start.elapsed();
        let shape = repr.embeddings.dims();

        println!("  Sample {}/{}", i + 1, args.num_samples);
        println!(
            "    Output shape: [{}, {}, {}]",
            shape[0], shape[1], shape[2]
        );
        println!(
            "    Inference time: {:.2}ms",
            elapsed.as_secs_f64() * 1000.0
        );

        let flat = repr.embeddings.reshape([shape[0] * shape[1], shape[2]]);
        let show_dims = shape[2].min(8);
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

fn run_safetensors(args: EncodeArgs) -> Result<()> {
    let vit_config = vit_config_for_preset(&args.preset);

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

    let checkpoint = load_encoder_checkpoint(&args.model)?;
    let tensor_map = checkpoint
        .tensors
        .iter()
        .map(|(key, tensor)| (key.clone(), tensor.to_tensor_data()))
        .collect();
    let encoder = vit_config
        .init::<B>(&DEVICE)
        .load_named_tensors(&tensor_map)
        .with_context(|| format!("failed to inject weights from {}", args.model.display()))?;

    println!(
        "  Loaded {} mapped tensor(s) from safetensors checkpoint.",
        checkpoint.tensors.len()
    );
    if !checkpoint.unmapped_keys.is_empty() {
        println!(
            "  Ignored {} unmapped checkpoint key(s).",
            checkpoint.unmapped_keys.len()
        );
    }
    println!();

    run_encoder_samples(
        &encoder,
        args.height,
        args.width,
        args.num_samples,
        embed_dim,
    )
}

fn run_demo(args: EncodeArgs) -> Result<()> {
    let vit_config = vit_config_for_preset(&args.preset);

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
    println!("  is only implemented for .safetensors and .onnx inputs).");
    println!();

    run_encoder_samples(
        &encoder,
        args.height,
        args.width,
        args.num_samples,
        embed_dim,
    )
}

fn run_encoder_samples(
    encoder: &jepa_vision::vit::VitEncoder<B>,
    height: usize,
    width: usize,
    num_samples: usize,
    embed_dim: usize,
) -> Result<()> {
    for i in 0..num_samples {
        let input: Tensor<B, 4> = Tensor::random(
            [1, 3, height, width],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &DEVICE,
        );

        let repr = encoder.encode(&input);
        let shape = repr.embeddings.dims();

        println!("  Sample {}/{}", i + 1, num_samples);
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

fn vit_config_for_preset(preset: &ArchPreset) -> VitConfig {
    match preset {
        ArchPreset::VitBase16 => VitConfig::vit_base_patch16(),
        ArchPreset::VitSmall16 => VitConfig::vit_small_patch16(),
        ArchPreset::VitLarge16 => VitConfig::vit_large_patch16(),
        ArchPreset::VitHuge14 => VitConfig::vit_huge_patch14(),
    }
}

#[derive(Debug)]
struct DemoInput {
    label: String,
    tensor: Tensor<B, 4>,
}

fn build_demo_inputs(height: usize, width: usize, sample_count: usize) -> Result<Vec<DemoInput>> {
    let mut inputs = Vec::with_capacity(sample_count);

    for (index, (relative_path, image)) in demo::demo_pattern_images()
        .into_iter()
        .take(sample_count)
        .enumerate()
    {
        inputs.push(DemoInput {
            label: format_demo_sample_label(index, &relative_path),
            tensor: demo_image_to_tensor(&image, height, width),
        });
    }

    Ok(inputs)
}

fn format_demo_sample_label(index: usize, relative_path: &str) -> String {
    let stem = std::path::Path::new(relative_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("sample");
    let readable = stem.replace('_', "-");
    format!("{:02} {}", index + 1, readable)
}

fn demo_image_to_tensor(image: &RgbImage, height: usize, width: usize) -> Tensor<B, 4> {
    let resized = resize(image, width as u32, height as u32, FilterType::Triangle);
    let data = rgb_image_to_chw(&resized);
    Tensor::from_floats(TensorData::new(data, [1, 3, height, width]), &DEVICE)
}

fn rgb_image_to_chw(image: &RgbImage) -> Vec<f32> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut data = vec![0.0f32; 3 * height * width];

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x as u32, y as u32).0;
            for channel in 0..3 {
                let value = f32::from(pixel[channel]) / 255.0;
                let normalized = (value - DEFAULT_RGB_MEAN[channel]) / DEFAULT_RGB_STD[channel];
                let index = channel * height * width + y * width + x;
                data[index] = normalized;
            }
        }
    }

    data
}

fn summarize_inference_sample(
    repr: jepa_core::Representation<B>,
    sample_label: &str,
    sample_index: usize,
    total_samples: usize,
    inference_time_ms: f64,
) -> Result<InferenceSampleMetrics> {
    let shape = repr.embeddings.dims();
    let embed_dim = shape[2];
    let values: Vec<f32> = repr
        .embeddings
        .to_data()
        .to_vec()
        .map_err(|e| anyhow::anyhow!("failed to read tensor data: {e}"))?;
    let (embedding_mean, embedding_std, mean_token_l2_norm) = embedding_summary(&values, embed_dim);

    Ok(InferenceSampleMetrics {
        sample_index,
        total_samples,
        sample_label: sample_label.to_string(),
        output_shape: [shape[0], shape[1], shape[2]],
        inference_time_ms,
        first_token_preview: values.iter().take(embed_dim.min(8)).copied().collect(),
        embedding_mean,
        embedding_std,
        mean_token_l2_norm,
        mask_present: repr.mask.is_some(),
    })
}

fn embedding_summary(values: &[f32], embed_dim: usize) -> (f64, f64, f64) {
    if values.is_empty() || embed_dim == 0 {
        return (0.0, 0.0, 0.0);
    }

    let len = values.len() as f64;
    let mean = values.iter().map(|value| f64::from(*value)).sum::<f64>() / len;
    let variance = values
        .iter()
        .map(|value| {
            let centered = f64::from(*value) - mean;
            centered * centered
        })
        .sum::<f64>()
        / len;
    let token_count = values.len() / embed_dim;
    let mean_token_l2_norm = if token_count == 0 {
        0.0
    } else {
        values
            .chunks(embed_dim)
            .map(|token| {
                token
                    .iter()
                    .map(|value| {
                        let value = f64::from(*value);
                        value * value
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .sum::<f64>()
            / token_count as f64
    };

    (mean, variance.sqrt(), mean_token_l2_norm)
}

fn load_encoder_checkpoint(path: &std::path::Path) -> Result<Checkpoint> {
    let mapped = jepa_compat::safetensors::load_checkpoint(path, &ijepa_vit_keymap())
        .context("failed to load safetensors checkpoint with I-JEPA keymap")?;
    if !mapped.is_empty() {
        return Ok(mapped);
    }

    jepa_compat::safetensors::load_raw_checkpoint(path)
        .context("failed to load raw safetensors checkpoint without key remapping")
}

#[cfg(test)]
mod tests {
    use super::embedding_summary;

    #[test]
    fn embedding_summary_reports_mean_std_and_token_norm() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let (mean, std, mean_token_norm) = embedding_summary(&values, 2);

        assert!((mean - 2.5).abs() < 1e-9);
        assert!((std - 1.118_033_988_749_895).abs() < 1e-9);

        let expected_norm = ((1.0f64.powi(2) + 2.0f64.powi(2)).sqrt()
            + (3.0f64.powi(2) + 4.0f64.powi(2)).sqrt())
            / 2.0;
        assert!((mean_token_norm - expected_norm).abs() < 1e-9);
    }

    #[test]
    fn embedding_summary_handles_empty_inputs() {
        let (mean, std, norm) = embedding_summary(&[], 0);
        assert_eq!((mean, std, norm), (0.0, 0.0, 0.0));
    }
}
