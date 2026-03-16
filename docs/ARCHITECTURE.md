# Architecture

## Project Identity

`jepa-rs` is a Rust workspace for JEPA research and tooling. It is primarily a
library plus local-tooling project, not a long-running service.

Shipped surfaces today:

- Reusable JEPA library crates built on `burn`
- A CLI and TUI in `crates/jepa`
- A browser demo crate in `crates/jepa-web`

## Workspace Map

| Crate | Role | Depends on |
|-------|------|------------|
| `jepa-core` | Shared contracts, masking, semantic tensor wrappers, energy, EMA, config | Workspace deps only |
| `jepa-vision` | ViT encoders, I-JEPA, V-JEPA, slot attention | `jepa-core` |
| `jepa-world` | Planning, memory, hierarchical/world-model helpers | `jepa-core` |
| `jepa-train` | Generic training loops, schedules, checkpoint metadata | `jepa-core` |
| `jepa-compat` | Safetensors, ONNX metadata/runtime, key remapping, model registry | `jepa-core` |
| `jepa` | CLI, demos, and ratatui dashboard | All library crates |
| `jepa-web` | Browser demo crate with WASM exports and JS assets | `jepa-core`, `jepa-vision`, `jepa-train` |

`fuzz/` is a separate nightly-only workspace for libFuzzer targets.

## Data Flow

### Strict image and video JEPA

The semantic reference path lives in `jepa-vision`:

1. Patch or tubelet embedding converts the raw input into positioned tokens.
2. Masking produces disjoint context and target index sets.
3. The context encoder sees only visible tokens before self-attention.
4. The target encoder produces target representations on the detached EMA path.
5. The predictor conditions on context embeddings and target positions.
6. Energy plus collapse regularization compute the loss.

The strict path matters because post-encoder slicing can silently leak target
information through attention. For that reason:

- `IJepa::forward_step_strict` is the reference image path.
- `VJepa::forward_step_strict` is the reference video path.
- `JepaComponents::forward_step` in `jepa-train` remains an approximate,
  cheaper helper and is documented that way.

### Training orchestration

`jepa-train` owns schedules, metrics, and generic forward-step orchestration.
It intentionally does not redefine vision semantics. If a change is about mask
placement, token positions, or parity with reference implementations, the fix
belongs in `jepa-vision`.

### Checkpoint and ONNX interop

`jepa-compat` owns external model format handling:

- safetensors loading and key remapping
- ONNX metadata parsing and initializer extraction
- tract-based ONNX execution helpers

Format-specific code stays here so the model crates do not leak loader or
runtime concerns into their public APIs.

### Browser demo

`jepa-web` reuses the model crates but keeps the current exported browser path
simple:

- The exported WASM API uses the deterministic `burn-ndarray` CPU backend.
- `burn-wgpu` remains linked for future runtime-selection work.
- Browser-visible config and tensor shapes are validated at the WASM boundary.

The browser demo is for local experimentation, not a validated production WebGPU
deployment target today.

## Key Invariants

- Public tensor-bearing APIs should prefer semantic wrappers such as
  `Representation<B>`, `Energy<B>`, and `Action<B>`.
- Configs must be validated before callers rely on them.
- Strict parity fixtures are the truth source for image-path semantic regressions.
- Library crates return typed errors for caller-triggerable failures.
- CLI and TUI boundaries use contextual `anyhow` errors.
- The browser demo maps typed internal validation errors to string errors at the
  WASM boundary because that is the JS-facing interface.

## Boundaries And Ownership

- `jepa-core` is the shared contract layer and is treated as gated.
- `Cargo.toml`, `Cargo.lock`, CI, scripts, and parity fixtures are gated paths.
- `crates/jepa-web` is autonomous for code and asset changes, but its manifest
  is still gated like other workspace manifests.

## Verification Expectations

Use the smallest useful proof first, then widen the scope:

- Crate-local change: run the owning crate tests.
- Cross-crate or public-behavior change: run workspace check, test, clippy, and
  format gates.
- Strict image semantic change: also run `scripts/run_parity_suite.sh`.

See [QUALITY_GATES.md](./QUALITY_GATES.md) for the exact command set.

## Operational Notes

- This repo does not expose a network service, so service-style health checks
  and trace IDs are out of scope.
- Production readiness here means truthful interfaces, reproducible verification,
  stable library contracts, and documented release behavior.
- The current blocker register lives in [PRODUCTION_GAPS.md](./PRODUCTION_GAPS.md).
