# Contributing to jepa-rs

Thank you for your interest in contributing to jepa-rs!

## Getting Started

```bash
git clone https://github.com/AbdelStark/jepa-rs
cd jepa-rs
cargo build
cargo test
```

## Development Workflow

1. **Build**: `cargo build`
2. **Workspace check**: `cargo check --workspace --all-targets`
3. **Test**: `cargo test --workspace`
4. **Targeted browser-demo test**: `cargo test -p jepa-web` when touching `crates/jepa-web`
5. **Lint**: `cargo clippy --workspace --all-targets -- -D warnings`
6. **Format**: `cargo fmt` (check with `cargo fmt -- --check`)
7. **Docs**: `cargo doc --workspace --no-deps`
8. **Coverage**: `cargo llvm-cov --workspace --all-features --fail-under-lines 80`
9. **Fuzz smoke**: run the commands from [`docs/QUALITY_GATES.md`](./docs/QUALITY_GATES.md)
10. **Bench smoke**: `cargo bench --workspace --no-run`

All release-blocking checks must pass before submitting a PR.

## Architecture

The project is organized as a Cargo workspace with 7 crates:

| Crate | Purpose |
|-------|---------|
| `jepa-core` | Core traits and types: `Encoder`, `Predictor`, `EnergyFn`, `MaskingStrategy`, `EMA` |
| `jepa-vision` | Vision Transformer (ViT), patch embedding, RoPE, I-JEPA, V-JEPA |
| `jepa-world` | World model, action conditioning, CEM planner, H-JEPA, memory |
| `jepa-train` | Training loop, LR schedulers, checkpointing |
| `jepa-compat` | PyTorch/safetensors weight loading, ONNX metadata inspection |
| `jepa` | CLI binary and interactive TUI dashboard |
| `jepa-web` | Browser demo crate; exported WASM path uses the CPU backend today |

All crates depend on `jepa-core`. There are no circular dependencies.

## Key Design Principles

- **Backend-agnostic**: All ML types are generic over `B: Backend` (burn's backend trait)
- **Semantic types**: Use `Representation<B>`, `Energy<B>` wrappers — never raw tensors in public APIs
- **Validate early**: All inputs validated at construction time
- **Typed misuse paths first**: Use `Result<T, E>` with `thiserror` error enums for caller-triggerable failures
- **Deterministic**: Use `rand_chacha` seeded RNG for reproducibility

## Writing Tests

- Place tests in `#[cfg(test)] mod tests` at the bottom of the source file
- Use `type TestBackend = burn_ndarray::NdArray<f32>;` for all tests
- Use `proptest` for numerical invariants and property-based testing
- Use `1e-6` tolerance for standard f32 comparisons, `1e-4` for chained operations

## Commit Convention

Format: `type(scope): description`

- **Types**: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
- **Scopes**: `core`, `vision`, `world`, `train`, `compat`, `cli`, `web`, `docs`, `specs`
- Example: `feat(core): implement EnergyFn trait with L2 and cosine variants`

## Reference

- [CHANGELOG.md](./CHANGELOG.md) — Version history
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) — System layout, invariants, and boundaries
- [docs/QUALITY_GATES.md](./docs/QUALITY_GATES.md) — Verification runbook
- [docs/RELEASE.md](./docs/RELEASE.md) — Release checklist and policy
- [docs/ROADMAP.md](./docs/ROADMAP.md) — Next milestones and exit criteria
- [docs/PRODUCTION_GAPS.md](./docs/PRODUCTION_GAPS.md) — Remaining blockers and risks
- [docs/agentic/project-decisions.md](./docs/agentic/project-decisions.md) — Durable architectural decisions
