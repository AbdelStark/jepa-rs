# jepa-rs

Rust workspace for JEPA building blocks on top of `burn` 0.20.1.

[Specification](./SPECIFICATION.md) | [Architecture](./ARCHITECTURE.md) | [Gap Analysis](./PRODUCTION_GAP.md) | [Roadmap](./ROADMAP.md) | [Work Packages](./WORK_PACKAGES.md) | [Contributing](./CONTRIBUTING.md) | [Changelog](./CHANGELOG.md)

## What This Is

`jepa-rs` implements the main components of Joint Embedding Predictive Architectures: core traits, vision encoders and predictors, world-model utilities, training helpers, checkpoint compatibility code, and parser-backed ONNX inspection/loading.

## Who It Is For

This repository is for Rust developers who want to experiment with JEPA-style models, extend the architecture, or build backend-agnostic ML components without depending on Python at runtime.

## Status

As of March 12, 2026, this project is **alpha with local release-candidate rehearsal complete**.

It is suitable for local research, API exploration, and extending JEPA components inside Rust codebases. It is not yet production-grade.

Known limitations:

- The generic trainer in [`crates/jepa-train/src/trainer.rs`](./crates/jepa-train/src/trainer.rs) still slices context and target tokens after encoder forward. Strict pre-attention masking is available through [`IJepa::forward_step_strict`](./crates/jepa-vision/src/image.rs) and [`VJepa::forward_step_strict`](./crates/jepa-vision/src/video.rs).
- Differential parity now runs in CI against three checked-in strict image fixtures. Broader modality coverage, especially strict video parity, is still pending.
- ONNX support covers model metadata and initializer loading, not full runtime execution.
- Workspace crates are not published to crates.io yet. Local package smoke checks pass, but consumers still depend on the git workspace until the first crates.io release lands.

## Workspace Layout

```text
jepa-core     Core traits, tensor wrappers, masking, energy, collapse regularization, EMA
jepa-vision   ViT encoder, image/video JEPA models, predictor implementations
jepa-world    Action-conditioned prediction, planning, hierarchy, short-term memory
jepa-train    Training-step orchestration, schedules, checkpoint metadata
jepa-compat   safetensors loading, key remapping, ONNX API surface
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for design notes and invariants, [PRODUCTION_GAP.md](./PRODUCTION_GAP.md) for the current blocker register, [ROADMAP.md](./ROADMAP.md) for milestone sequencing, and [WORK_PACKAGES.md](./WORK_PACKAGES.md) for implementation-sized tasks.
Operational guidance for verification and release lives in [docs/QUALITY_GATES.md](./docs/QUALITY_GATES.md), [docs/RELEASE.md](./docs/RELEASE.md), [docs/OPERATIONS.md](./docs/OPERATIONS.md), and [docs/PERFORMANCE.md](./docs/PERFORMANCE.md).

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
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt -- --check
cargo doc --workspace --no-deps
scripts/run_parity_suite.sh
```

`scripts/run_parity_suite.sh` runs every checked-in fixture in [`specs/differential`](./specs/differential) by default.

Extended quality gates:

```bash
cargo llvm-cov --workspace --all-features --fail-under-lines 80
(cd fuzz && cargo fuzz run masking -- -runs=1000)
cargo bench --workspace --no-run
```

Target a single crate when iterating:

```bash
cargo test -p jepa-core
cargo test -p jepa-vision
```

## Support Boundary

- Strict image and video helpers are the semantic reference paths for masked JEPA training behavior.
- The generic trainer is reusable orchestration, not a faithful masked-encoder reference implementation.
- ONNX support stops at metadata inspection and initializer loading. Executing ONNX graphs is out of scope unless maintainers approve a separate expansion.
- Release and troubleshooting runbooks are documented for external users in [`docs/OPERATIONS.md`](./docs/OPERATIONS.md) and [`docs/RELEASE.md`](./docs/RELEASE.md).


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

### Suggested use inside `jepa-rs`

- For image-path semantics and future differential tests, start with `facebookresearch/ijepa`.
- For video-path semantics, predictor wiring, and checkpoint naming, use `facebookresearch/jepa` and `facebookresearch/vjepa2`.
- For smaller end-to-end examples spanning representation learning through planning, compare against `facebookresearch/eb_jepa`.
- Treat the papers above as the source of truth for intended JEPA behavior when documentation and existing code disagree.
