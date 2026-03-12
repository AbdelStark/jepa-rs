<p align="center">
  <h1 align="center">jepa-rs</h1>
  <p align="center">
    <strong>Joint Embedding Predictive Architecture in Rust</strong>
  </p>
  <p align="center">
    <a href="https://github.com/AbdelStark/jepa-rs/actions"><img src="https://img.shields.io/github/actions/workflow/status/AbdelStark/jepa-rs/ci.yml?branch=main&style=flat-square&logo=github&label=CI" alt="CI"></a>
    <a href="https://github.com/AbdelStark/jepa-rs/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="License: MIT"></a>
    <a href="https://docs.rs/jepa-core"><img src="https://img.shields.io/badge/docs-docs.rs-blue.svg?style=flat-square" alt="docs.rs"></a>
    <a href="https://crates.io/crates/jepa-core"><img src="https://img.shields.io/crates/v/jepa-core.svg?style=flat-square" alt="crates.io"></a>
  </p>
</p>

---

Production-quality Rust implementation of **JEPA** (Joint Embedding Predictive Architecture) — the self-supervised learning framework from [Yann LeCun and Meta AI](https://openreview.net/pdf?id=BZ5a1r-kVsf) for learning world models that predict in representation space rather than pixel space.

**jepa-rs** provides modular, backend-agnostic building blocks for I-JEPA (images), V-JEPA (video), and hierarchical world models, built on top of the [burn](https://burn.dev) deep learning framework. It includes full ONNX runtime support for inference with pretrained Facebook Research models.

```
                    ┌──────────────┐
                    │   Context    │──── Encoder ────┐
                    │   (visible)  │                 │
   Image/Video ────┤              │         ┌───────▼───────┐
                    │   Target     │         │   Predictor   │──── predicted repr
                    │   (masked)   │──┐      └───────────────┘          │
                    └──────────────┘  │                                 │
                                      │      ┌───────────────┐         │
                                      └──────│ Target Encoder│── target repr
                                        EMA  │   (frozen)    │         │
                                              └───────────────┘         │
                                                                        │
                                              ┌───────────────┐         │
                                              │  Energy Loss  │◄────────┘
                                              └───────────────┘
```

## Why jepa-rs?

| | jepa-rs | Python (PyTorch) |
|---|---|---|
| **Runtime** | Native binary, no Python/CUDA dependency | Requires Python + PyTorch + CUDA |
| **Inference** | ONNX via tract (CPU/GPU), zero-copy | PyTorch runtime |
| **Memory** | Rust ownership, no GC pauses | Python GC + PyTorch allocator |
| **Backend** | Any burn backend (CPU, GPU, WebGPU, WASM) | CUDA-centric |
| **Type safety** | Compile-time tensor shape checks | Runtime shape errors |
| **Deployment** | Single static binary | Docker + Python environment |

## Pretrained Models

jepa-rs supports loading official Facebook Research pretrained JEPA models:

| Model | Architecture | Params | Resolution | Dataset | Weights |
|-------|-------------|--------|-----------|---------|---------|
| **I-JEPA ViT-H/14** | ViT-Huge, patch 14 | 632M | 224x224 | ImageNet-1K | [Download](https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar) \| [HuggingFace](https://huggingface.co/facebook/ijepa_vith14_1k) |
| **I-JEPA ViT-H/16-448** | ViT-Huge, patch 16 | 632M | 448x448 | ImageNet-1K | [Download](https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16.448-300e.pth.tar) \| [HuggingFace](https://huggingface.co/facebook/ijepa_vith16_448) |
| **I-JEPA ViT-H/14** | ViT-Huge, patch 14 | 632M | 224x224 | ImageNet-22K | [Download](https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar) |
| **I-JEPA ViT-G/16** | ViT-Giant, patch 16 | 1.0B | 224x224 | ImageNet-22K | [Download](https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar) |
| **V-JEPA ViT-L/16** | ViT-Large, patch 16 | 304M | 224x224 | VideoMix2M | [Download](https://dl.fbaipublicfiles.com/jepa/vit.l.16-k400-300e.pth.tar) |
| **V-JEPA ViT-H/16** | ViT-Huge, patch 16 | 632M | 224x224 | VideoMix2M | [Download](https://dl.fbaipublicfiles.com/jepa/vit.h.16-k400-300e.pth.tar) |

## Quick Start

### Installation

```toml
# Cargo.toml
[dependencies]
jepa-core   = { git = "https://github.com/AbdelStark/jepa-rs" }
jepa-vision = { git = "https://github.com/AbdelStark/jepa-rs" }
jepa-compat = { git = "https://github.com/AbdelStark/jepa-rs" }  # For ONNX + checkpoint loading
```

### ONNX Inference with Pretrained Models

The fastest path to running a real JEPA model:

```bash
# 1. Export a pretrained I-JEPA model to ONNX (requires Python + PyTorch)
pip install torch onnx
python scripts/export_ijepa_onnx.py --model vit_h14

# 2. Run inference in Rust
cargo run -p jepa-compat --example onnx_inference -- ijepa_vit_h14_encoder.onnx
```

Or create a tiny test model (no GPU needed):

```bash
python scripts/export_ijepa_onnx.py --tiny-test
cargo run -p jepa-compat --example onnx_inference -- ijepa_tiny_test.onnx
```

### ONNX Inference from Rust

```rust
use jepa_compat::runtime::OnnxSession;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load an exported ONNX model
    let session = OnnxSession::from_path("ijepa_vit_h14_encoder.onnx")?;
    println!("Model: {:?}", session.info());

    // Prepare input: [batch=1, channels=3, height=224, width=224]
    let input = vec![0.0f32; 1 * 3 * 224 * 224];
    let output = session.run_f32(&[1, 3, 224, 224], &input)?;

    // Output: [1, 256, 1280] — 256 patch tokens, 1280-dim embeddings
    if let Some((data, tokens, embed_dim)) = output.as_token_embeddings() {
        println!("Got {} token embeddings of dim {}", tokens, embed_dim);
    }
    Ok(())
}
```

### Loading SafeTensors Checkpoints

```rust
use jepa_compat::safetensors::load_checkpoint;
use jepa_compat::keymap::ijepa_vit_keymap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mappings = ijepa_vit_keymap();
    let checkpoint = load_checkpoint("model.safetensors", &mappings)?;

    println!("Loaded {} tensors", checkpoint.len());
    for key in checkpoint.keys() {
        println!("  {}: {:?}", key, checkpoint.get(key).unwrap().shape);
    }
    Ok(())
}
```

### Building JEPA Models from Scratch

```rust
use burn::prelude::*;
use burn_ndarray::NdArray;
use jepa_core::masking::{BlockMasking, MaskingStrategy};
use jepa_core::types::InputShape;
use jepa_vision::image::IJepaConfig;
use jepa_vision::vit::VitConfig;

type B = NdArray<f32>;

fn main() {
    let device = burn_ndarray::NdArrayDevice::Cpu;

    // Configure I-JEPA with ViT-Huge/14 (matches Facebook pretrained)
    let config = IJepaConfig {
        encoder: VitConfig::vit_huge_patch14(),
        predictor: jepa_vision::image::TransformerPredictorConfig {
            encoder_embed_dim: 1280,
            predictor_embed_dim: 384,
            num_layers: 12,
            num_heads: 12,
            max_target_len: 256,
        },
    };
    let model = config.init::<B>(&device);

    // Generate masks (I-JEPA block masking)
    let shape = InputShape::Image { height: 16, width: 16 }; // 224/14 = 16
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let masking = BlockMasking {
        num_targets: 4,
        target_scale: (0.15, 0.2),
        target_aspect_ratio: (0.75, 1.5),
    };
    let mask = masking.generate_mask(&shape, &mut rng);

    println!("Context tokens: {}, Target tokens: {}",
             mask.context_indices.len(), mask.target_indices.len());
}
```

### Browse Available Models

```rust
use jepa_compat::registry::{list_models, find_model};

fn main() {
    for model in list_models() {
        println!("{}: {} ({}, {})",
            model.name,
            model.param_count_human(),
            model.architecture,
            model.pretrained_on);
    }

    // Search for a specific model
    if let Some(m) = find_model("vit-h/14") {
        println!("\nFound: {} with {} patches",
            m.name, m.num_patches());
    }
}
```

## Architecture

```text
jepa-rs/
├── jepa-core        Core traits, tensor wrappers, masking, energy, EMA
│   ├── Encoder          Trait for context/target encoders
│   ├── Predictor        Trait for latent predictors
│   ├── EnergyFn         L2, Cosine, SmoothL1 energy functions
│   ├── MaskingStrategy  Block, MultiBlock, Spatiotemporal masking
│   ├── CollapseReg      VICReg, BarlowTwins collapse prevention
│   └── EMA              Exponential moving average with cosine schedule
│
├── jepa-vision      Vision transformers and JEPA models
│   ├── VitEncoder       ViT-S/B/L/H/G with 2D RoPE
│   ├── IJepa            I-JEPA pipeline (image)
│   ├── VJepa            V-JEPA pipeline (video, 3D tubelets)
│   └── Predictor        Transformer-based cross-attention predictor
│
├── jepa-world       World models and planning
│   ├── ActionPredictor  Action-conditioned latent prediction
│   ├── Planner          Random shooting planner with cost functions
│   ├── HierarchicalJepa Multi-level H-JEPA
│   └── ShortTermMemory  Sliding-window memory for temporal context
│
├── jepa-train       Training orchestration
│   ├── TrainConfig      Learning rate schedules, EMA config
│   ├── JepaComponents   Generic forward step orchestration
│   └── CheckpointMeta   Save/resume metadata
│
└── jepa-compat      Model compatibility and ONNX runtime
    ├── OnnxSession      Full ONNX inference via tract engine
    ├── ModelRegistry     Pretrained model catalog (Facebook Research)
    ├── SafeTensors       Load .safetensors checkpoints
    ├── KeyMap            PyTorch → burn key remapping
    └── OnnxModelInfo     ONNX metadata inspection
```

All tensor-bearing APIs are generic over `B: Backend`, allowing transparent execution on CPU (NdArray), GPU (WGPU), or WebAssembly backends.

## ONNX Runtime Support

jepa-rs includes a complete ONNX inference pipeline powered by [tract](https://github.com/sonos/tract):

```text
PyTorch Model (.pth)
        │
        ▼  scripts/export_ijepa_onnx.py
  ONNX Model (.onnx)
        │
        ▼  jepa_compat::runtime::OnnxSession
  Rust Inference (tract)
        │
        ▼
  Token Embeddings [B, N, D]
```

### Supported operations

- **Model loading**: Parse and optimize ONNX graphs
- **Shape inference**: Automatic or manual input shape specification
- **Execution**: Full forward pass through transformer encoder
- **Output extraction**: Token embeddings as flat f32 arrays
- **Model inspection**: Metadata, input/output specs, validation diagnostics

### Export workflow

```bash
# Install Python dependencies
pip install torch onnx

# Export ViT-H/14 (downloads ~2.5GB checkpoint automatically)
python scripts/export_ijepa_onnx.py --model vit_h14

# Export ViT-G/16 (1B parameter model)
python scripts/export_ijepa_onnx.py --model vit_g16

# Verify with onnxruntime
python scripts/export_ijepa_onnx.py --model vit_h14 --verify

# Create tiny model for testing (no downloads needed)
python scripts/export_ijepa_onnx.py --tiny-test
```

## Examples

| Example | Description | Run command |
|---------|-------------|-------------|
| `ijepa_demo` | Full I-JEPA forward pass pipeline | `cargo run -p jepa-vision --example ijepa_demo` |
| `ijepa_train_loop` | Training loop with metrics | `cargo run -p jepa-vision --example ijepa_train_loop` |
| `world_model_planning` | World model with random shooting | `cargo run -p jepa-world --example world_model_planning` |
| `onnx_inference` | ONNX model inference | `cargo run -p jepa-compat --example onnx_inference -- model.onnx` |
| `model_registry` | Browse pretrained models | `cargo run -p jepa-compat --example model_registry` |

## Build & Test

```bash
# Build everything
cargo build --workspace

# Run all tests
cargo test --workspace

# Lint
cargo clippy --workspace --all-targets -- -D warnings

# Format check
cargo fmt -- --check

# Generate docs
cargo doc --workspace --no-deps --open

# Run differential parity tests
scripts/run_parity_suite.sh

# Target a single crate
cargo test -p jepa-core
cargo test -p jepa-vision
cargo test -p jepa-compat
```

### Extended quality gates

```bash
# Code coverage (requires cargo-llvm-cov)
cargo llvm-cov --workspace --all-features --fail-under-lines 80

# Fuzz testing (requires cargo-fuzz)
(cd fuzz && cargo fuzz run masking -- -runs=1000)

# Benchmark smoke test
cargo bench --workspace --no-run
```

## Project Status

**Alpha** — suitable for research, experimentation, and extension.

### What works

- Complete I-JEPA and V-JEPA architectures with strict masked-encoder paths
- Full ONNX inference runtime for pretrained Facebook Research models
- SafeTensors checkpoint loading with automatic key remapping
- Pretrained model registry with download URLs
- Differential parity tests against 3 checked-in strict image fixtures
- Comprehensive test suite (400+ tests), property-based testing, fuzz targets
- All standard ViT configs: ViT-S/16, ViT-B/16, ViT-L/16, ViT-H/14, ViT-H/16, ViT-G/16

### Known limitations

- The generic trainer slices tokens after encoder forward; strict pre-attention masking is available via `IJepa::forward_step_strict` and `VJepa::forward_step_strict`
- Differential parity runs in CI for strict image fixtures; broader video parity is pending
- Workspace crates not yet published to crates.io (depends on git for now)

## References

### Papers

| Paper | Focus |
|-------|-------|
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) | JEPA position paper (LeCun, 2022) |
| [I-JEPA](https://arxiv.org/abs/2301.08243) | Self-supervised image learning (Assran et al., 2023) |
| [V-JEPA](https://openreview.net/forum?id=WFYbBOEOtv) | Latent video prediction (Bardes et al., 2024) |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | Video understanding + planning (Bardes et al., 2025) |
| [EB-JEPA](https://arxiv.org/abs/2602.03604) | Lightweight JEPA library (2026) |

### Official implementations

| Repo | Models |
|------|--------|
| [`facebookresearch/ijepa`](https://github.com/facebookresearch/ijepa) | I-JEPA (archived) |
| [`facebookresearch/jepa`](https://github.com/facebookresearch/jepa) | V-JEPA |
| [`facebookresearch/vjepa2`](https://github.com/facebookresearch/vjepa2) | V-JEPA 2 |
| [`facebookresearch/eb_jepa`](https://github.com/facebookresearch/eb_jepa) | EB-JEPA tutorial |

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](./LICENSE) for details.

---

<p align="center">
  <sub>Built with <a href="https://burn.dev">burn</a> and <a href="https://github.com/sonos/tract">tract</a></sub>
</p>
