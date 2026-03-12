# jepa-rs

Rust workspace for JEPA building blocks on top of `burn` 0.16.

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
