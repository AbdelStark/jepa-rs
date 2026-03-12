# jepa-rs

Rust workspace for JEPA building blocks on top of `burn` 0.20.1.

[Specification](./SPECIFICATION.md) | [Architecture](./ARCHITECTURE.md) | [Gap Analysis](./PRODUCTION_GAP.md) | [Roadmap](./ROADMAP.md) | [Work Packages](./WORK_PACKAGES.md) | [Contributing](./CONTRIBUTING.md) | [Changelog](./CHANGELOG.md)

## What This Is

`jepa-rs` implements the main components of Joint Embedding Predictive Architectures: core traits, vision encoders and predictors, world-model utilities, training helpers, and checkpoint compatibility code.

## Who It Is For

This repository is for Rust developers who want to experiment with JEPA-style models, extend the architecture, or build backend-agnostic ML components without depending on Python at runtime.

## Status

As of March 12, 2026, this project is **alpha**.

It is suitable for local research, API exploration, and extending JEPA components inside Rust codebases. It is not yet suitable for parity-sensitive production training or deployment pipelines.

Known limitations:

- The generic trainer in [`crates/jepa-train/src/trainer.rs`](./crates/jepa-train/src/trainer.rs) slices context and target tokens after encoder forward, so it does not enforce strict pre-encoder masking semantics.
- ONNX loading in [`crates/jepa-compat/src/onnx.rs`](./crates/jepa-compat/src/onnx.rs) is still a stub until an ONNX runtime dependency is added.
- Differential tests against the Python reference implementations are not wired yet.
- Fuzz targets and CI-enforced coverage thresholds are not present yet.
- Workspace crates are not published to crates.io yet.

## Workspace Layout

```text
jepa-core     Core traits, tensor wrappers, masking, energy, collapse regularization, EMA
jepa-vision   ViT encoder, image/video JEPA models, predictor implementations
jepa-world    Action-conditioned prediction, planning, hierarchy, short-term memory
jepa-train    Training-step orchestration, schedules, checkpoint metadata
jepa-compat   safetensors loading, key remapping, ONNX API surface
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for design notes and invariants, [PRODUCTION_GAP.md](./PRODUCTION_GAP.md) for the current blocker register, [ROADMAP.md](./ROADMAP.md) for milestone sequencing, and [WORK_PACKAGES.md](./WORK_PACKAGES.md) for implementation-sized tasks.

## Important JEPA Resources

These are the main external references to use when you need the original algorithm papers, official PyTorch implementations, or smaller codebases that are easier to compare against while working on `jepa-rs`.

### Foundational papers

| Resource | Focus | Why it matters here |
| --- | --- | --- |
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) | JEPA / H-JEPA position paper | Best high-level description of the world-model agenda behind JEPA. |
| [Introduction to Latent Variable Energy-Based Models: A Path Towards Autonomous Machine Intelligence](https://arxiv.org/abs/2306.02572) | Tutorial-style JEPA / EBM overview | Good conceptual bridge between energy-based models, latent variables, and H-JEPA. |

### Core model papers

| Resource | Focus | Why it matters here |
| --- | --- | --- |
| [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) | I-JEPA | Canonical reference for image masking, target blocks, EMA target encoders, and frozen evaluation. |
| [V-JEPA: Latent Video Prediction for Visual Representation Learning](https://openreview.net/forum?id=WFYbBOEOtv) | V-JEPA | Canonical reference for spatiotemporal masking and latent prediction from video. |
| [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985) | V-JEPA 2 | Most relevant paper here for action-conditioned world modeling and planning. |
| [Intuitive physics understanding emerges from self-supervised pretraining on natural videos](https://arxiv.org/abs/2502.11831) | JEPA evaluation / probing | Useful downstream evidence for why JEPA-style video representations matter beyond benchmark accuracy. |
| [A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures](https://arxiv.org/abs/2602.03604) | EB-JEPA | Compact 2026 reference library covering image, video, and action-conditioned JEPA examples. |

### Official open-source implementations

| Repo | Focus | Why it matters here |
| --- | --- | --- |
| [`facebookresearch/ijepa`](https://github.com/facebookresearch/ijepa) | I-JEPA | Archived official image reference; best parity target for image masking behavior and checkpoint layout. |
| [`facebookresearch/jepa`](https://github.com/facebookresearch/jepa) | V-JEPA | Official video training stack; useful for video masking, predictor contracts, and pretrained checkpoints. |
| [`facebookresearch/vjepa2`](https://github.com/facebookresearch/vjepa2) | V-JEPA 2 | Official follow-on codebase with newer video models and action-conditioned components. |
| [`facebookresearch/jepa-intuitive-physics`](https://github.com/facebookresearch/jepa-intuitive-physics) | JEPA evaluation | Reproducible downstream evaluation code built on top of the V-JEPA stack. |
| [`facebookresearch/eb_jepa`](https://github.com/facebookresearch/eb_jepa) | EB-JEPA tutorial library | Small official examples spanning image JEPA, video JEPA, and action-conditioned planning. |

### Smaller community implementations

| Repo | Focus | Why it matters here |
| --- | --- | --- |
| [`filipbasara0/simple-ijepa`](https://github.com/filipbasara0/simple-ijepa) | Minimal I-JEPA | Easier to read end to end than the full Meta training stack. |
| [`LumenPallidium/jepa`](https://github.com/LumenPallidium/jepa) | Experimental JEPA playground | Useful for exploratory variants and implementation notes, but not a canonical reference. |

### Suggested use inside `jepa-rs`

- For image-path semantics and future differential tests, start with `facebookresearch/ijepa`.
- For video-path semantics, predictor wiring, and checkpoint naming, use `facebookresearch/jepa` and `facebookresearch/vjepa2`.
- For smaller end-to-end examples spanning representation learning through planning, compare against `facebookresearch/eb_jepa`.
- Treat the papers above as the source of truth for intended JEPA behavior when documentation and existing code disagree.

## Quick Start

The crates are not published yet, so depend on the workspace directly from git:

```toml
[dependencies]
jepa-core = { git = "https://github.com/AbdelStark/jepa-rs" }
jepa-vision = { git = "https://github.com/AbdelStark/jepa-rs" }
```

Minimal masking example:

```rust
use jepa_core::masking::{BlockMasking, MaskingStrategy};
use jepa_core::types::InputShape;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() {
    let masking = BlockMasking {
        num_targets: 4,
        target_scale: (0.15, 0.2),
        target_aspect_ratio: (0.75, 1.5),
    };

    let shape = InputShape::Image { height: 14, width: 14 };
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mask = masking.generate_mask(&shape, &mut rng);

    assert!(mask.validate().is_ok());
    assert_eq!(mask.context_indices.len() + mask.target_indices.len(), 196);
}
```

Runnable examples live in [`crates/jepa-vision/examples`](./crates/jepa-vision/examples) and [`crates/jepa-world/examples`](./crates/jepa-world/examples).

## Build And Test

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt -- --check
cargo doc --no-deps
```

Target a single crate when iterating:

```bash
cargo test -p jepa-core
cargo test -p jepa-vision
```

## Development Notes

- `SPECIFICATION.md` is the design source of truth. Treat it as read-only unless a human explicitly asks to change it.
- Public trait signatures and `Cargo.toml` changes are gated work.
- `Representation::gather` preserves masks and should be preferred over ad hoc token slicing.
- `TransformerPredictor` expects `target_positions` to contain real flattened token indices, not placeholder zeros.

## Planning Docs

- [PRODUCTION_GAP.md](./PRODUCTION_GAP.md): current blocker register and dependency graph
- [ROADMAP.md](./ROADMAP.md): milestone order and exit criteria
- [WORK_PACKAGES.md](./WORK_PACKAGES.md): implementation-ready task breakdown

## Help

Open a GitHub issue in the repository for bugs, missing features, or design questions.
