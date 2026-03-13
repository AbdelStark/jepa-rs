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

Alpha Rust implementation of **JEPA** (Joint Embedding Predictive Architecture) — the self-supervised learning framework from [Yann LeCun and Meta AI](https://openreview.net/pdf?id=BZ5a1r-kVsf) for learning world models that predict in representation space rather than pixel space.

**jepa-rs** provides modular, backend-agnostic building blocks for I-JEPA (images), V-JEPA (video), and hierarchical world models, built on top of the [burn](https://burn.dev) deep learning framework. It includes a CLI and interactive TUI dashboard, safetensors checkpoint loading, ONNX metadata inspection, and a pretrained model registry for Facebook Research models.

```
                    ┌──────────────┐
                    │   Context    │──── Encoder ────┐
                    │   (visible)  │                 │
   Image/Video ─────┤              │         ┌───────▼───────┐
                    │   Target     │         │   Predictor   │──── predicted repr
                    │   (masked)   │──┐      └───────────────┘          │
                    └──────────────┘  │                                 │
                                      │      ┌───────────────┐          │
                                      └──────│ Target Encoder│── target repr
                                        EMA  │   (frozen)    │          │
                                             └───────────────┘          │
                                                                        │
                                             ┌───────────────┐          │
                                             │  Energy Loss  │◄─────────┘
                                             └───────────────┘
```

## Why jepa-rs?

| | jepa-rs | Python (PyTorch) |
|---|---|---|
| **Runtime** | Native binary, no Python/CUDA dependency | Requires Python + PyTorch + CUDA |
| **Inference** | Safetensors checkpoint loading, ONNX metadata | PyTorch runtime |
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
jepa-core   = "0.1.0"
jepa-vision = "0.1.0"
jepa-compat = "0.1.0"  # For ONNX + checkpoint loading
```

### CLI

The `jepa` binary provides a unified CLI for the workspace:

```bash
# Install the CLI from crates.io
cargo install jepa

# Or install from the local workspace checkout
cargo install --path crates/jepa

# Launch the interactive TUI dashboard
jepa

# List pretrained models in the registry
jepa models

# Inspect a safetensors checkpoint
jepa inspect model.safetensors

# Analyze checkpoint with key remapping
jepa checkpoint model.safetensors --keymap ijepa --verbose

# Launch a training run
jepa train --preset vit-base-16 --steps 10 --batch-size 1 --lr 1e-3

# Train from a normal image directory tree with deterministic resize/crop/normalize
jepa train --preset vit-base-16 --steps 100 --batch-size 4 \
  --dataset-dir ./images/train --resize 256 --crop-size 224 --shuffle

# Train from a safetensors image tensor dataset [N, C, H, W]
jepa train --preset vit-base-16 --steps 100 --batch-size 1 \
  --dataset train.safetensors --dataset-key images

# Encode inputs through a safetensors checkpoint
jepa encode --model model.safetensors --preset vit-base-16

# Or through an ONNX model
jepa encode --model model.onnx --height 224 --width 224
```

The CLI `train` command now runs real strict masked-image optimization with
AdamW and EMA. It chooses one input source per run:

- `--dataset-dir <PATH>` for a recursive image-folder dataset (`jpg`, `jpeg`, `png`, `webp`) with decode, RGB conversion, shorter-side resize, center crop, CHW tensor conversion, and normalization
- `--dataset <FILE> --dataset-key <KEY>` for a safetensors image tensor shaped `[N, C, H, W]`
- no dataset flags for the synthetic random-tensor fallback

Image-folder preprocessing defaults to the preset image size for `--crop-size`
and the ImageNet RGB normalization statistics when `--mean` and `--std` are
omitted. Dataset loading is currently single-threaded.
`jepa encode` executes real encoder weights for `.safetensors` and `.onnx`
inputs; other extensions still fall back to the preset demo path.

### Runnable Examples

The `jepa` crate now ships runnable examples under
`crates/jepa/examples/` that exercise the real training command instead of
mocking the CLI path:

```bash
# Create a tiny recursive image-folder dataset under target/example-data/jepa/
cargo run -p jepa --example prepare_demo_image_folder

# Train for 2 steps on that generated image-folder dataset
cargo run -p jepa --example train_image_folder_demo

# Train for 2 steps with the synthetic fallback path
cargo run -p jepa --example train_synthetic_demo
```

The image-folder example deliberately uses a very small generated dataset
(6 PNG files across nested subdirectories). That is enough for a meaningful
smoke demo of recursive dataset discovery, decode, resize, crop, normalize,
batching, masking, optimizer updates, and EMA without checking a large image
corpus into git. It is not large enough to demonstrate real representation
learning quality; it is an execution demo, not a benchmark dataset.

The TUI now incorporates these demos in the `Training` tab as a guided demo
runner. Launch `jepa`, switch to tab `3`, choose a demo with `j/k`, and press
`Enter` to run it. The panel streams real run logs, step metrics, loss/energy
charts, and a short interpretation of what happened.

The TUI `Inference` tab on `4` adds a separate guided walkthrough for encoder
inference. It runs deterministic demo image patterns through a preset ViT,
streams phase changes, per-sample latency and embedding statistics, and explains
what the representation telemetry means. The walkthrough is intentionally a
pipeline demo rather than a pretrained semantic benchmark.

If you want to run the CLI directly after generating the demo dataset:

```bash
cargo run -p jepa -- train --preset vit-small-16 --steps 2 --batch-size 2 \
  --dataset-dir target/example-data/jepa/demo-image-folder \
  --resize 256 --crop-size 224 --shuffle --dataset-limit 6
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
├── jepa-compat      Model compatibility and interop
│   ├── ModelRegistry     Pretrained model catalog (Facebook Research)
│   ├── SafeTensors       Load .safetensors checkpoints
│   ├── KeyMap            PyTorch → burn key remapping
│   └── OnnxModelInfo     ONNX metadata inspection and initializer loading
│
└── jepa             CLI and interactive TUI dashboard
    ├── CLI               models, inspect, checkpoint, train, encode commands
    └── TUI               Dashboard, Models, Training, Checkpoint, About tabs
```

All tensor-bearing APIs are generic over `B: Backend`, allowing transparent execution on CPU (NdArray), GPU (WGPU), or WebAssembly backends.

## ONNX Support

jepa-rs provides ONNX metadata inspection and initializer loading through `jepa-compat`. This allows inspecting model structure, input/output specs, and importing weight initializers from `.onnx` files.

**Current scope**: metadata inspection and weight import are production-ready. Tract-based ONNX graph execution exists (`OnnxSession`, `OnnxEncoder`) but is not yet production-grade — it is functional for prototyping and testing.

## Examples

| Example | Description | Run command |
|---------|-------------|-------------|
| `jepa` | Interactive TUI dashboard | `cargo run -p jepa` |
| `jepa models` | Browse pretrained model registry | `cargo run -p jepa -- models` |
| `jepa train` | Launch a training run | `cargo run -p jepa -- train --preset vit-base-16` |
| `prepare_demo_image_folder` | Generate a tiny recursive dataset for `--dataset-dir` demos | `cargo run -p jepa --example prepare_demo_image_folder` |
| `train_image_folder_demo` | Run the real `jepa train` image-folder path on generated images | `cargo run -p jepa --example train_image_folder_demo` |
| `train_synthetic_demo` | Run the real `jepa train` synthetic fallback path | `cargo run -p jepa --example train_synthetic_demo` |
| `ijepa_demo` | Full I-JEPA forward pass pipeline | `cargo run -p jepa-vision --example ijepa_demo` |
| `ijepa_train_loop` | Training loop with metrics | `cargo run -p jepa-vision --example ijepa_train_loop` |
| `world_model_planning` | World model with random shooting | `cargo run -p jepa-world --example world_model_planning` |
| `model_registry` | Browse pretrained models (library) | `cargo run -p jepa-compat --example model_registry` |

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
- CLI with 6 commands (`models`, `inspect`, `checkpoint`, `train`, `encode`, `tui`)
- Interactive TUI dashboard with 6 tabs (Dashboard, Models, Training, Inference, Checkpoint, About)
- SafeTensors checkpoint loading with automatic key remapping
- ONNX metadata inspection and initializer loading
- Pretrained model registry with download URLs
- Differential parity tests against 3 checked-in strict image fixtures
- Comprehensive test suite (365 tests), property-based testing, fuzz targets
- All standard ViT configs: ViT-S/16, ViT-B/16, ViT-L/16, ViT-H/14, ViT-H/16, ViT-G/16

### Known limitations

- The generic trainer slices tokens after encoder forward; strict pre-attention masking is available via `IJepa::forward_step_strict` and `VJepa::forward_step_strict`
- ONNX support covers metadata inspection and initializer loading only, not graph execution
- Differential parity runs in CI for strict image fixtures; broader video parity is pending
- First-time crates.io release must be published in dependency order because the workspace crates depend on each other by version

## JEPA Variants: What We Implement

The JEPA family has grown across several papers. Here is exactly what jepa-rs implements and how each component maps to a specific paper and reference codebase.

### I-JEPA (Image)

| | |
|---|---|
| **Paper** | [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) (Assran et al., CVPR 2023) |
| **Reference code** | [`facebookresearch/ijepa`](https://github.com/facebookresearch/ijepa) (archived) |
| **jepa-rs struct** | `IJepa<B>` in `jepa-vision` ([`crates/jepa-vision/src/image.rs`](crates/jepa-vision/src/image.rs)) |
| **What it does** | Self-supervised image representation learning. A ViT context-encoder sees only visible patches; a lightweight predictor predicts representations of masked target patches. The target-encoder is an EMA copy of the context-encoder. |
| **Masking** | `BlockMasking` — contiguous rectangular blocks on the 2D patch grid. |
| **Faithful path** | `IJepa::forward_step_strict` — filters tokens *before* encoder self-attention (matches the paper). |
| **Approximate path** | `JepaComponents::forward_step` in `jepa-train` — encodes full input then slices (post-encoder masking; cheaper but not faithful). |
| **Parity status** | 3 checked-in strict image fixtures verified in CI. |

### V-JEPA (Video)

| | |
|---|---|
| **Paper** | [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471) (Bardes et al., 2024) |
| **Reference code** | [`facebookresearch/jepa`](https://github.com/facebookresearch/jepa) |
| **jepa-rs struct** | `VJepa<B>` in `jepa-vision` ([`crates/jepa-vision/src/video.rs`](crates/jepa-vision/src/video.rs)) |
| **What it does** | Extends I-JEPA to video. A ViT encoder processes 3D tubelets (space + time) with 3D RoPE. |
| **Masking** | `SpatiotemporalMasking` — contiguous 3D regions in the spatiotemporal grid. |
| **Faithful path** | `VJepa::forward_step_strict` — pre-attention masking. |
| **Parity status** | Implemented but strict video parity not yet proven (pending). |

### V-JEPA 2 features

| | |
|---|---|
| **Paper** | [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985) (Bardes et al., 2025) |
| **Reference code** | [`facebookresearch/vjepa2`](https://github.com/facebookresearch/vjepa2) |
| **jepa-rs support** | Not a separate struct. The `VJepa<B>` struct can be configured with V-JEPA 2 features. |
| **What we take from V-JEPA 2** | **Cosine momentum schedule** for EMA — `CosineMomentumSchedule` in `jepa-core` (`Ema::with_cosine_schedule`). Momentum ramps from base (e.g. 0.996) to 1.0 over training. Also: `MultiBlockMasking` strategy, ViT-Giant/14 preset. |
| **What we don't implement** | The full V-JEPA 2 training recipe, attentive probing, or the planning/action heads from the paper. |

### Hierarchical JEPA (H-JEPA) — experimental

| | |
|---|---|
| **Paper** | Inspired by [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) (LeCun, 2022) — the original JEPA position paper describes hierarchical prediction as a long-term goal. No standalone H-JEPA paper exists yet. |
| **jepa-rs struct** | `HierarchicalJepa<B>` in `jepa-world` ([`crates/jepa-world/src/hierarchy.rs`](crates/jepa-world/src/hierarchy.rs)) |
| **What it does** | Stacks multiple JEPA levels at different temporal strides (e.g. stride 2, 6, 24). Each level has its own encoder and predictor. This is **experimental** — no reference implementation exists. |

### Action-Conditioned World Model — experimental

| | |
|---|---|
| **Paper** | Draws from both the LeCun position paper and [V-JEPA 2](https://arxiv.org/abs/2506.09985) (planning component). |
| **jepa-rs structs** | `Action<B>`, `ActionConditionedPredictor<B>` trait, `RandomShootingPlanner` in `jepa-world` ([`crates/jepa-world/src/action.rs`](crates/jepa-world/src/action.rs), [`crates/jepa-world/src/planner.rs`](crates/jepa-world/src/planner.rs)) |
| **What it does** | Predicts next-state representations given current state + action. Supports random-shooting (CEM) planning. This is **experimental**. |

### What about EB-JEPA?

[EB-JEPA](https://arxiv.org/abs/2602.03604) (Terver et al., 2026) is a separate lightweight Python library for energy-based JEPA. jepa-rs is **not** an implementation of EB-JEPA. We reference it for comparison only. The energy functions in `jepa-core` (L2, Cosine, SmoothL1) are standard loss formulations, not the EB-JEPA energy framework.

### Quick summary

| Variant | Paper | jepa-rs struct | Status |
|---------|-------|----------------|--------|
| I-JEPA | Assran et al. 2023 | `IJepa<B>` | Strict path implemented, parity verified |
| V-JEPA | Bardes et al. 2024 | `VJepa<B>` | Strict path implemented, parity pending |
| V-JEPA 2 | Bardes et al. 2025 | `VJepa<B>` + cosine EMA schedule | Select features only |
| H-JEPA | LeCun 2022 (position paper) | `HierarchicalJepa<B>` | Experimental, no reference impl |
| World model | LeCun 2022 + V-JEPA 2 | `ActionConditionedPredictor`, `RandomShootingPlanner` | Experimental |
| EB-JEPA | Terver et al. 2026 | **Not implemented** | Referenced for comparison only |

## References

### Papers

| Paper | Focus |
|-------|-------|
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) | JEPA position paper — hierarchical world models (LeCun, 2022) |
| [I-JEPA](https://arxiv.org/abs/2301.08243) | Self-supervised image learning with masked prediction in latent space (Assran et al., CVPR 2023) |
| [V-JEPA](https://arxiv.org/abs/2404.08471) | Extension to video with spatiotemporal masking (Bardes et al., 2024) |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | Video understanding, prediction, and planning (Bardes et al., 2025) |
| [EB-JEPA](https://arxiv.org/abs/2602.03604) | Lightweight energy-based JEPA library — referenced for comparison (Terver et al., 2026) |

### Official reference implementations

| Repo | Models | Relationship to jepa-rs |
|------|--------|------------------------|
| [`facebookresearch/ijepa`](https://github.com/facebookresearch/ijepa) | I-JEPA (archived) | Primary reference for `IJepa<B>` and key remapping |
| [`facebookresearch/jepa`](https://github.com/facebookresearch/jepa) | V-JEPA | Primary reference for `VJepa<B>` |
| [`facebookresearch/vjepa2`](https://github.com/facebookresearch/vjepa2) | V-JEPA 2 | Reference for cosine EMA schedule, ViT-G config |
| [`facebookresearch/eb_jepa`](https://github.com/facebookresearch/eb_jepa) | EB-JEPA tutorial | Not implemented — comparison only |

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](./LICENSE) for details.

---

<p align="center">
  <sub>Built with <a href="https://burn.dev">burn</a> and <a href="https://github.com/sonos/tract">tract</a></sub>
</p>
