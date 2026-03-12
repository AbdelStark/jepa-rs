<identity>
jepa-rs is an alpha Rust workspace for JEPA components built on burn 0.16.

As of March 12, 2026, the workspace compiles cleanly and its current tests pass, safetensors support is functional, ONNX loading is still a stub, and the generic training helper still slices tokens after encoder forward instead of enforcing strict pre-encoder masking.
</identity>

<architecture>

| Path | Responsibility |
|------|----------------|
| `crates/jepa-core` | Shared tensor wrappers, traits, masking, energy, collapse prevention, EMA, config |
| `crates/jepa-vision` | ViT encoders, image/video JEPA pieces, predictor implementations |
| `crates/jepa-world` | Action-conditioned rollout, planning, hierarchy, short-term memory |
| `crates/jepa-train` | Training-step orchestration, schedules, checkpoint metadata |
| `crates/jepa-compat` | safetensors loading, key remapping, ONNX API surface |

Read [`ARCHITECTURE.md`](./ARCHITECTURE.md) before making architectural changes.
Read [`PRODUCTION_GAP.md`](./PRODUCTION_GAP.md), [`ROADMAP.md`](./ROADMAP.md), and [`WORK_PACKAGES.md`](./WORK_PACKAGES.md) before planning substantial work.
</architecture>

<stack>

| Layer | Tooling |
|-------|---------|
| Language | Rust 2021 |
| Tensor backend | `burn = 0.16` |
| CPU backend in tests | `burn-ndarray = 0.16` |
| Serialization | `serde = 1`, `serde_json = 1` |
| Errors | `thiserror = 2` |
| Checkpoint format | `safetensors = 0.4` |
| Testing | `cargo test`, `proptest`, crate-local integration tests |
| Linting | `cargo clippy --all-targets -- -D warnings` |
| Formatting | `cargo fmt -- --check` |

</stack>

<commands>

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt -- --check
cargo doc --no-deps
```

Useful focused runs:

```bash
cargo test -p jepa-core
cargo test -p jepa-vision
cargo test -p jepa-train
cargo test -p jepa-compat
```

</commands>

<conventions>

- Keep tensor-bearing public APIs generic over `B: Backend`.
- Prefer semantic wrappers like `Representation<B>` and `Energy<B>` over raw tensors in public interfaces.
- Validate invalid user input with typed errors where the existing API allows it.
- Treat panics as invariant checks, not ordinary runtime error handling.
- Add tests for every behavioral fix. Regression tests matter more than broad refactors.
- Use concise comments for non-obvious invariants or correctness constraints only.

</conventions>

<critical_constraints>

- Do not modify `SPECIFICATION.md` without explicit human approval.
- Do not modify workspace or crate `Cargo.toml` files without explicit human approval.
- Do not change existing public trait signatures without explicit human approval.
- Do not claim ONNX loading works. It does not yet.
- Do not describe the generic trainer as a faithful masked JEPA trainer without qualifying the current limitation.

</critical_constraints>

<gotchas>

- `JepaComponents::forward_step` is a generic orchestration helper. It validates masks and passes real target indices, but it still filters tokens after encoder forward because `Encoder::Input` is opaque.
- `TransformerPredictor` expects `target_positions` to contain real flattened token indices.
- `Representation::gather` preserves masks. If a downstream change drops masks again, that is a regression.
- `OnnxModelInfo::from_file` distinguishes missing files from runtime-unavailable errors, but successful parsing still requires a future runtime dependency.

</gotchas>

<current_state>

- Verified locally: build, tests, clippy, and docs.
- Missing: strict masked-encoder training path, ONNX runtime integration, differential tests, fuzz targets, crates.io release packaging.
- Project status: alpha library for experimentation and extension, not production-ready parity code.
- Active planning source of truth: `PRODUCTION_GAP.md`, `ROADMAP.md`, `WORK_PACKAGES.md`.

</current_state>
