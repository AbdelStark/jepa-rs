# Architecture

## Project Identity

`jepa-rs` is an alpha Rust workspace that implements JEPA-oriented model components on top of the `burn` tensor stack.

The repository is a library workspace, not a deployed service. "Production readiness" here means API clarity, numerical correctness, reproducible local verification, and honest documentation of what is still missing.

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
- ONNX metadata API surface

Today, safetensors support is functional and ONNX loading is not.

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

### Strict masked-encoder semantics are incomplete

The generic path in [`crates/jepa-train/src/trainer.rs`](./crates/jepa-train/src/trainer.rs) encodes the full input and then slices visible and target tokens. That keeps downstream shapes honest, but it does not stop encoder self-attention from seeing target tokens. A stricter training path needs encoder-specific masked-input support.

### ONNX is an adapter stub

[`crates/jepa-compat/src/onnx.rs`](./crates/jepa-compat/src/onnx.rs) validates file existence and exposes error types, but it does not parse ONNX graphs yet.

### Reference parity is not proven

The workspace has a substantial unit and integration suite, but it still lacks differential checks against the canonical Python JEPA implementations.

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
```

## Current State

As of March 12, 2026:

- Workspace build, test, clippy, and docs pass locally.
- Safetensors loading is usable.
- ONNX runtime integration is not implemented.
- The generic trainer remains an approximation of strict JEPA masking semantics.
- Differential tests, fuzz targets, and crates.io release work remain open.

## Next Three Milestones

### 1. Masked Encoder Correctness

Scope:

- add encoder-specific masked forward paths
- add regression tests that prove hidden tokens cannot influence context encoding

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

- ONNX runtime support
- fuzz targets for masking and energy code
- release packaging

Exit criteria:

- ONNX integration is functional, not stubbed
- fuzz targets run in CI or dedicated automation
- crates are publishable with accurate release notes
