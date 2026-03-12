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
2. **Test**: `cargo test` (runs 267 unit/integration + 23 doc tests)
3. **Lint**: `cargo clippy --all-targets`
4. **Format**: `cargo fmt` (check with `cargo fmt -- --check`)
5. **Docs**: `cargo doc --no-deps`

All five checks must pass before submitting a PR.

## Architecture

The project is organized as a Cargo workspace with 5 crates:

| Crate | Purpose |
|-------|---------|
| `jepa-core` | Core traits and types: `Encoder`, `Predictor`, `EnergyFn`, `MaskingStrategy`, `EMA` |
| `jepa-vision` | Vision Transformer (ViT), patch embedding, RoPE, I-JEPA, V-JEPA |
| `jepa-world` | World model, action conditioning, CEM planner, H-JEPA, memory |
| `jepa-train` | Training loop, LR schedulers, checkpointing |
| `jepa-compat` | PyTorch/safetensors weight loading, ONNX import |

All crates depend on `jepa-core`. There are no circular dependencies.

## Key Design Principles

- **Backend-agnostic**: All ML types are generic over `B: Backend` (burn's backend trait)
- **Semantic types**: Use `Representation<B>`, `Energy<B>` wrappers — never raw tensors in public APIs
- **Validate early**: All inputs validated at construction time
- **No unwrap in library code**: Use `Result<T, E>` with `thiserror` error enums
- **Deterministic**: Use `rand_chacha` seeded RNG for reproducibility

## Writing Tests

- Place tests in `#[cfg(test)] mod tests` at the bottom of the source file
- Use `type TestBackend = burn_ndarray::NdArray<f32>;` for all tests
- Use `proptest` for numerical invariants and property-based testing
- Use `1e-6` tolerance for standard f32 comparisons, `1e-4` for chained operations

## Commit Convention

Format: `type(scope): description`

- **Types**: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
- **Scopes**: `core`, `vision`, `world`, `train`, `compat`, `specs`
- Example: `feat(core): implement EnergyFn trait with L2 and cosine variants`

## Reference

- [SPECIFICATION.md](./SPECIFICATION.md) — RFC archive (read-only, source of truth)
- [CHANGELOG.md](./CHANGELOG.md) — Version history
- [specs/gherkin/features.feature](./specs/gherkin/features.feature) — BDD scenarios
