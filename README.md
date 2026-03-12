# jepa-rs

**Production-grade Rust implementation of the Joint Embedding Predictive Architecture (JEPA).**

[![License](https://img.shields.io/badge/license-Apache--2.0%2FMIT-blue)](#license)

[Specification](./SPECIFICATION.md) | [BDD Features](./specs/gherkin/features.feature) | [Changelog](./CHANGELOG.md)

---

## What is JEPA?

[JEPA (Joint Embedding Predictive Architecture)](https://openreview.net/pdf?id=BZ5a1r-kVsf) is a self-supervised learning framework proposed by Yann LeCun. Unlike generative models that predict in pixel space, JEPA predicts in **representation space** — learning abstract, semantic features without reconstruction.

JEPA is the architecture behind [I-JEPA](https://github.com/facebookresearch/ijepa) (images), [V-JEPA](https://github.com/facebookresearch/jepa) (video), and [V-JEPA 2](https://ai.meta.com/vjepa/), and is the core technology of [AMI Labs](https://amilabs.xyz/).

## Why Rust?

All existing JEPA implementations are Python/PyTorch. `jepa-rs` brings JEPA to Rust for:

- **Safety-critical deployment** — memory safety guarantees for healthcare, robotics, and industrial settings
- **Deterministic execution** — no garbage collector, no runtime surprises; prerequisite for verifiable AI
- **Bare-metal inference** — run on ARM, RISC-V, wearables, and edge devices without a Python runtime
- **Production infrastructure** — video preprocessing, data pipelines, and model serving at scale

## Architecture

```
jepa-core     Core traits: Encoder, Predictor, EnergyFn, MaskingStrategy, EMA
jepa-vision   Vision Transformer (ViT), patchification, RoPE, I-JEPA, V-JEPA
jepa-world    Action conditioning, CEM planner, H-JEPA, memory
jepa-train    Training loop, LR schedulers, checkpointing
jepa-compat   Load PyTorch/safetensors weights, ONNX import
```

Backend-agnostic via the [burn](https://burn.dev) framework. Supports CPU (`ndarray`), GPU (`wgpu`, `cuda`), and WASM.

## Quick Start

Add `jepa-rs` crates to your `Cargo.toml`:

```toml
[dependencies]
jepa-core = { git = "https://github.com/AbdelStark/jepa-rs" }
jepa-vision = { git = "https://github.com/AbdelStark/jepa-rs" }
```

### Example: I-JEPA Forward Pass

```rust
use burn::prelude::*;
use burn_ndarray::NdArray;
use jepa_core::masking::{BlockMasking, MaskingStrategy};
use jepa_core::types::InputShape;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

type B = NdArray<f32>;

fn main() {
    // Define masking strategy (I-JEPA style)
    let masking = BlockMasking {
        num_targets: 4,
        target_scale: (0.15, 0.2),
        target_aspect_ratio: (0.75, 1.5),
    };

    // Generate a mask for a 14x14 patch grid (ViT-Base with 16x16 patches on 224x224 images)
    let shape = InputShape::Image { height: 14, width: 14 };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mask = masking.generate_mask(&shape, &mut rng);

    assert!(mask.validate().is_ok());
    println!("Context tokens: {}, Target tokens: {}",
        mask.context_indices.len(), mask.target_indices.len());
}
```

See [`crates/jepa-vision/examples/`](./crates/jepa-vision/examples/) for complete I-JEPA training loop examples.

## Status

**v0.1.0-dev** — All 10 RFCs from [SPECIFICATION.md](./SPECIFICATION.md) are implemented across the workspace.

| Component | Status | Tests |
|-----------|--------|-------|
| Core traits & types | Complete | 93 unit + property tests |
| Vision (ViT, I-JEPA, V-JEPA) | Complete | 38 unit + property + BDD tests |
| World model & planning | Complete | 23 unit + property tests |
| Training loop | Complete | 21 unit + property tests |
| PyTorch compat (safetensors) | Complete | 31 unit + property tests |
| ONNX runtime | API-complete (stub) | 8 tests |

### Remaining work for v0.1.0

- ONNX runtime integration (needs `ort` crate dependency)
- Differential testing against Python reference implementations
- Published crate on crates.io

## Development

```bash
# Build
cargo build

# Run all tests (245 unit/integration + 16 doc tests)
cargo test

# Run tests for a specific crate
cargo test -p jepa-core

# Lint
cargo clippy --all-targets

# Format
cargo fmt -- --check

# Benchmarks
cargo bench -p jepa-core
cargo bench -p jepa-vision
```

## Reference Implementations

Differential tests are designed to run against these Python codebases:

| Repo | Description |
|------|-------------|
| [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) | I-JEPA (images) |
| [facebookresearch/jepa](https://github.com/facebookresearch/jepa) | V-JEPA / V-JEPA 2 (video) |
| [facebookresearch/eb_jepa](https://github.com/facebookresearch/eb_jepa) | EB-JEPA (educational library) |
| [facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms) | JEPA World Models |

## License

Dual-licensed under [Apache-2.0](./LICENSE-APACHE) or [MIT](./LICENSE-MIT) at your option.

## Author

[Abdel Bakhta](https://github.com/AbdelStark) ([@AbdelStark](https://x.com/AbdelStark))
