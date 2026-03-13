# Architecture

## Project Identity

`jepa-rs` is an alpha Rust workspace that implements JEPA-oriented model components on top of the `burn` tensor stack.

The repository is a library workspace with a CLI/TUI binary, not a deployed service. "Production readiness" here means API clarity, numerical correctness, reproducible local verification, and honest documentation of what is still missing.

Execution planning lives in:

- [PRODUCTION_GAP.md](./PRODUCTION_GAP.md)
- [ROADMAP.md](./ROADMAP.md)
- [WORK_PACKAGES.md](./WORK_PACKAGES.md)

## Crate Boundaries

### `jepa-core`

Owns the shared vocabulary:

- `Representation<B>` and `Energy<B>` wrappers
- `Encoder`, `Predictor`, `EnergyFn`, `MaskingStrategy`, `CollapseRegularizer`
- masking strategies, EMA utilities, configuration types

This crate is the most sensitivity-heavy boundary. Public trait changes here cascade through the rest of the workspace.

### `jepa-vision`

Implements the vision-specific JEPA pieces:

- image patch embedding and 2D RoPE
- video tubelet embedding and 3D RoPE
- ViT-style encoders
- `TransformerPredictor`
- `IJepa` and `VJepa` model shells

### `jepa-world`

Owns world-model utilities:

- `Action<B>`
- `ActionConditionedPredictor`
- rollout and random-shooting planning
- hierarchy and short-term memory

### `jepa-train`

Owns orchestration logic:

- training-step bookkeeping
- learning-rate schedules
- checkpoint metadata
- `JepaComponents::forward_step`

This crate does not own an optimizer or a full trainer runtime.

### `jepa-compat`

Owns interoperability:

- safetensors loading
- key remapping from Python checkpoints
- ONNX metadata inspection and initializer loading
- pretrained model registry

Safetensors support is functional. ONNX metadata inspection and initializer loading are implemented; full ONNX graph execution is not.

### `jepa`

Owns the user-facing binary:

- CLI with 6 subcommands: `models`, `inspect`, `checkpoint`, `train`, `encode`, `tui`
- interactive TUI dashboard with 5 tabs (Dashboard, Models, Training, Checkpoint, About)
- built on `clap` (CLI) and `ratatui`/`crossterm` (TUI)

This crate depends on all library crates but no library crate depends on it.

## Data Flow

### Vision JEPA

1. Raw image or video input is patchified or tubelet-embedded.
2. Positional information is added through RoPE.
3. A transformer encoder produces a sequence representation.
4. A predictor consumes context tokens plus target positions and emits predicted target embeddings.
5. An energy function compares predictions against target representations.
6. Collapse regularization adds a second training signal.

### World Model

1. A state representation and action sequence enter the planner.
2. The dynamics model rolls the state forward step by step.
3. A cost function scores the final trajectory relative to a goal.
4. The random-shooting planner refits a simple sampling distribution over candidate actions.

## Important Invariants

- `MaskSpec` partitions context and target indices without duplicates or out-of-bounds positions.
- `Representation::gather` must preserve both token order and any attached validity mask.
- `Predictor::predict` must treat `target_positions` as real flattened token indices.
- `JepaConfig` and `TrainConfig` reject invalid dimensions or momentum ranges at validation time.
- Workspace code must stay backend-generic over `B: Backend`.

## Known Gaps

### Generic training remains approximate

The generic path in [`crates/jepa-train/src/trainer.rs`](./crates/jepa-train/src/trainer.rs) still encodes the full input and then slices visible and target tokens. That keeps the orchestration reusable across backends and modalities, but it does not stop encoder self-attention from seeing target tokens.

Strict pre-attention masking is now provided through the concrete vision-model helpers:

- [`crates/jepa-vision/src/image.rs`](./crates/jepa-vision/src/image.rs) via `IJepa::forward_step_strict`
- [`crates/jepa-vision/src/video.rs`](./crates/jepa-vision/src/video.rs) via `VJepa::forward_step_strict`

### ONNX parsing and initializer loading are implemented

[`crates/jepa-compat/src/onnx.rs`](./crates/jepa-compat/src/onnx.rs) parses real ONNX `ModelProto` files, extracts input/output metadata, and loads embedded initializers into the checkpoint abstraction used by the rest of the workspace.

Current scope is model inspection and weight import, not general ONNX runtime execution.

### Reference parity covers strict image flows

Differential parity now runs in CI against three checked-in strict I-JEPA image fixtures. Strict video parity remains unproven.

## Verification Runbook

Run these commands from a clean checkout:

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt -- --check
cargo doc --no-deps
```

When editing numerical code, also run the relevant crate-specific tests first:

```bash
cargo test -p jepa-core
cargo test -p jepa-vision
cargo test -p jepa-train
cargo test -p jepa-compat
```

## Current State

As of March 13, 2026:

- Workspace build, test, clippy, and docs pass locally.
- Safetensors and ONNX initializer loading are usable.
- Strict image and video masked forward paths exist alongside the generic approximate trainer.
- Differential parity runs against three checked-in strict I-JEPA image fixtures in CI.
- Fuzz targets, coverage policy, and benchmark smoke checks are part of the verification surface.
- CLI binary (`jepa`) and interactive TUI dashboard are available.
- crates.io publication still requires follow-through.

## Next Three Milestones

### 1. Masked Encoder Correctness

Scope:

- keep strict image/video masked paths covered by no-leakage regression tests
- maintain clear separation between generic approximate orchestration and strict modality-specific paths

Exit criteria:

- no-leakage tests exist for image and video training paths
- generic trainer docs clearly distinguish approximate versus strict behavior

### 2. Reference Parity

Scope:

- add differential fixtures against at least one Python JEPA implementation
- document tolerated numeric error bounds

Exit criteria:

- parity suite runs locally and in CI
- at least one end-to-end image path is compared against a reference implementation

### 3. Interop And Hardening

Scope:

- ONNX metadata and checkpoint support
- fuzz targets for masking and energy code
- release packaging

Exit criteria:

- ONNX integration is functional for metadata and initializer loading
- fuzz targets run in CI or dedicated automation
- crates are publishable with accurate release notes
