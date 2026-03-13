<identity>
jepa-rs is an alpha Rust workspace for JEPA (Joint Embedding Predictive Architecture) components built on burn 0.20.1.

The workspace compiles cleanly, 356 tests pass, safetensors and ONNX metadata/initializer loading work, strict masked-encoder paths exist for image and video, and a CLI binary with interactive TUI dashboard is included.
</identity>

<architecture>

| Path | Responsibility |
|------|----------------|
| `crates/jepa-core` | Shared tensor wrappers (`Representation`, `Energy`), core traits (`Encoder`, `Predictor`, `EnergyFn`, `MaskingStrategy`, `CollapseRegularizer`), EMA, config |
| `crates/jepa-vision` | ViT encoders (image + video), patch embedding, RoPE, `TransformerPredictor`, `IJepa`/`VJepa` shells with strict paths |
| `crates/jepa-world` | Action-conditioned rollout, `RandomShootingPlanner` (CEM), `HierarchicalJepa`, `ShortTermMemory` |
| `crates/jepa-train` | `JepaComponents` forward-step orchestrator, LR schedules, checkpoint metadata (no optimizer — caller owns that) |
| `crates/jepa-compat` | safetensors loading, PyTorch key remapping, ONNX metadata/initializer inspection, ONNX runtime via tract |
| `crates/jepa` | CLI binary (6 subcommands) and interactive TUI dashboard (5 tabs) |

Read [`ARCHITECTURE.md`](./ARCHITECTURE.md) before making architectural changes.
Read [`PRODUCTION_GAP.md`](./PRODUCTION_GAP.md), [`ROADMAP.md`](./ROADMAP.md), and [`WORK_PACKAGES.md`](./WORK_PACKAGES.md) before planning substantial work.
</architecture>

<structure>
```
crates/
├── jepa-core/src/       # Core traits & types [create/modify]
│   ├── types.rs         # Representation<B>, Energy<B>, MaskSpec, InputShape
│   ├── encoder.rs       # Encoder<B> trait
│   ├── predictor.rs     # Predictor<B> trait
│   ├── energy.rs        # EnergyFn<B> + L2, Cosine, SmoothL1
│   ├── masking.rs       # MaskingStrategy + Block, Spatiotemporal, MultiBlock
│   ├── collapse.rs      # CollapseRegularizer<B> + VICReg, BarlowTwins
│   ├── ema.rs           # Ema, CosineMomentumSchedule
│   └── config.rs        # JepaConfig with builder + validation
├── jepa-vision/src/     # Vision-specific implementations [create/modify]
│   ├── vit.rs           # VitConfig presets (Tiny→Giant), VitEncoder<B>
│   ├── patch.rs         # PatchEmbedding (2D images)
│   ├── rope.rs          # RotaryPositionEncoding2D
│   ├── image.rs         # IJepa + forward_step_strict
│   └── video.rs         # VitVideoEncoder, VJepa + forward_step_strict
├── jepa-world/src/      # World model primitives [create/modify]
│   ├── action.rs        # Action<B>, ActionConditionedPredictor
│   ├── planner.rs       # WorldModel, RandomShootingPlanner, CostFunction
│   ├── hierarchy.rs     # HierarchicalJepa, JepaLevel
│   └── memory.rs        # ShortTermMemory (bounded FIFO)
├── jepa-train/src/      # Training orchestration [create/modify]
│   ├── trainer.rs       # JepaComponents::forward_step (generic orchestrator)
│   ├── step.rs          # TrainStepOutput, TrainConfig, TrainMetrics
│   ├── schedule.rs      # LrSchedule + WarmupCosine, Constant
│   └── checkpoint.rs    # CheckpointMeta
├── jepa-compat/src/     # Interop/loading [create/modify]
│   ├── safetensors.rs   # LoadedTensor, F16/BF16→f32 widening
│   ├── onnx.rs          # OnnxModelInfo, metadata + initializer inspection
│   ├── runtime.rs       # OnnxSession via tract, OnnxEncoder<B> adapter
│   ├── keymap.rs        # PyTorch → burn key remapping
│   └── registry.rs      # Pretrained model discovery
├── jepa/src/            # CLI binary + TUI [create/modify]
│   ├── cli.rs           # Cli struct, 6 subcommands
│   ├── commands/        # Command implementations
│   └── tui/             # Ratatui dashboard (5 tabs)
docs/                    # Project documentation [read-only unless updating docs]
fuzz/                    # Fuzz targets [create/modify]
scripts/                 # Build/CI scripts [read-only]
specs/                   # Gherkin BDD specs [read-only]
SPECIFICATION.md         # RFC archive [NEVER modify without approval]
ARCHITECTURE.md          # Crate boundaries [read before changing]
```
</structure>

<stack>

| Layer | Tooling | Version |
|-------|---------|---------|
| Language | Rust | 2021 edition, MSRV 1.85 |
| Tensor backend | burn | 0.20.1 (with autodiff feature) |
| CPU backend (tests) | burn-ndarray | 0.20.1 |
| GPU backend | burn-wgpu | 0.20.1 |
| Serialization | serde, serde_json | 1.x |
| Errors | thiserror | 2.x |
| Checkpoint format | safetensors | 0.7 |
| F16 widening | half | 2.x |
| CLI | clap | 4.x |
| TUI | ratatui + crossterm | 0.29 + 0.28 |
| Testing | cargo test, proptest 1.x | crate-local |
| Benchmarks | criterion | 0.8.2 |
| Linting | clippy | -D warnings |
| Formatting | rustfmt | --check |

</stack>

<commands>

| Task | Command | Notes |
|------|---------|-------|
| Build | `cargo build` | All 6 crates + binary |
| Test (all) | `cargo test --workspace` | 356 tests, ~30s |
| Test (crate) | `cargo test -p jepa-core` | Focused per-crate runs |
| Clippy | `cargo clippy --all-targets -- -D warnings` | Zero warnings required |
| Format check | `cargo fmt -- --check` | Must pass before commit |
| Docs | `cargo doc --no-deps` | Builds rustdoc |
| Single test | `cargo test -p jepa-core -- test_name` | Run one test |
| Test verbose | `cargo test -p jepa-core -- --nocapture` | See println output |

CI runs 11 jobs: check, test, clippy, fmt, doc, coverage (80% floor), bench-smoke, parity, package-smoke, fuzz, audit.

</commands>

<conventions>

- Keep tensor-bearing public APIs generic over `B: Backend`.
- Prefer semantic wrappers (`Representation<B>`, `Energy<B>`) over raw `Tensor<B, N>` in public interfaces.
- Use `thiserror::Error` for all error types. Propagate with `?`, never `unwrap()` in library code.
- Treat panics as invariant violations, not error handling. Document remaining panics.
- Add tests for every behavioral fix. Regression tests matter more than broad refactors.
- Use concise comments for non-obvious invariants only. No prose filler.
- Config types use builder pattern with `.validate()` returning `Result<T, ConfigError>`.
- Test backend: `type TestBackend = burn_ndarray::NdArray<f32>;` in all `#[cfg(test)]` modules.
- File naming: snake_case.rs. Type naming: PascalCase. Method naming: snake_case.
- Imports: `use crate::` for intra-crate, `use jepa_core::` for cross-crate.
- All public types derive `Debug, Clone`. Configs also derive `Serialize, Deserialize`.

</conventions>

<critical_constraints>

- Do not modify `SPECIFICATION.md` without explicit human approval.
- Do not modify workspace or crate `Cargo.toml` files without explicit human approval.
- Do not change existing public trait signatures without explicit human approval.
- Do not claim ONNX runtime execution works for production use. Metadata inspection and initializer loading are proven; tract-based execution exists but is not production-grade.
- Do not describe `JepaComponents::forward_step` as a faithful masked JEPA trainer. It masks after encoder forward (approximate). Use `IJepa::forward_step_strict` or `VJepa::forward_step_strict` for pre-attention masking.

</critical_constraints>

<gotchas>

- `JepaComponents::forward_step` encodes entire input then slices tokens (post-encoder masking). This is because `Encoder::Input` is an associated type — cannot mask before encoder without modality-specific helpers. Use the strict paths for faithful JEPA.
- `TransformerPredictor` expects `target_positions` to contain real flattened token indices, not offsets or masks.
- `Representation::gather` preserves masks. Any downstream change that drops masks is a regression — always verify mask flow.
- `OnnxModelInfo::from_file` distinguishes missing files from runtime-unavailable errors. Do not conflate these.
- `jepa-train` computes losses but does not own an optimizer — the caller manages optimization (burn convention).
- EMA momentum can be constant or cosine-scheduled (V-JEPA 2 style). Query via `.get_momentum(step)`.
- Config validation is mandatory: always call `.validate()` on `JepaConfig` and `TrainConfig` before use.

</gotchas>

<workflows>

<new_feature>
1. Read relevant ARCHITECTURE.md section and any applicable RFC in SPECIFICATION.md
2. Identify target crate and module. Check existing patterns in that crate.
3. Write tests first (TDD): `#[cfg(test)] mod tests` with `TestBackend = burn_ndarray::NdArray<f32>`
4. Implement until tests pass: `cargo test -p [crate]`
5. Run `cargo clippy --all-targets -- -D warnings` — zero warnings
6. Run `cargo fmt -- --check` — must pass
7. Run `cargo test --workspace` — no regressions in other crates
8. Self-review: check for mask preservation, correct tensor ranks, no unwrap in lib code
</new_feature>

<bug_fix>
1. Reproduce with a failing test first
2. Fix the root cause
3. Verify the test passes: `cargo test -p [crate] -- test_name`
4. Run full workspace tests to check for regressions
5. Run clippy and fmt
</bug_fix>

<adding_tests>
1. Use `type TestBackend = burn_ndarray::NdArray<f32>` and `NdArrayDevice::Cpu`
2. Test invariants, not implementation details
3. Use tolerances for float comparison (1e-6 single ops, 1e-4 chains)
4. Use proptest for numerical properties that should hold universally
5. Name tests: `test_[what]_[condition]_[expected]`
</adding_tests>

</workflows>

<boundaries>

<forbidden>
DO NOT modify under any circumstances:
- `.env`, `.env.*` — credentials/secrets
- `SPECIFICATION.md` — RFC archive (requires human approval)
</forbidden>

<gated>
Modify ONLY with explicit human approval:
- `Cargo.toml` (root or any crate) — dependency/version changes
- Public trait signatures in jepa-core — `Encoder`, `Predictor`, `EnergyFn`, `MaskingStrategy`, `CollapseRegularizer`
- `.github/workflows/ci.yml` — CI pipeline
- `scripts/` — build/deploy scripts
</gated>

<safety_checks>
Before any destructive operation:
1. State what you are about to do
2. State what could go wrong
3. Wait for confirmation
</safety_checks>

</boundaries>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| `the trait Backend is not implemented` | Missing generic bound or wrong backend type | Ensure function is generic over `B: Backend` |
| `expected Tensor<_, 3> found Tensor<_, 2>` | Rank mismatch from sum/squeeze/reshape | Check operation — some ops change tensor rank |
| `cannot move out of borrowed content` | Tensor consumed by operation | Use `.clone()` before the consuming operation |
| E0432 unresolved import after adding type | Missing `pub use` in crate's lib.rs | Add re-export to lib.rs |
| Float assertion fails intermittently | Tolerance too tight | Widen to 1e-5 or 1e-4 for accumulated operations |
| Slow compilation after changes | burn macro re-expansion | Expected on first build; subsequent builds are incremental |
| `cargo test` shows fewer tests than expected | Test filter active or compile error | Run `cargo test -p [crate]` without filter |

<recovery>
1. Read the FIRST compiler error — later errors cascade from it
2. Use `cargo check` for faster feedback (no codegen)
3. Run `RUST_BACKTRACE=1 cargo test` for runtime panics
4. Check proptest-regressions/ if a proptest fails — fix the bug, don't delete the file
5. If stuck, state the problem clearly and ask for help
</recovery>

</troubleshooting>

<skills>
Modular skills in `.codex/skills/` (symlinked at `.claude/skills/` and `.agents/skills/`).

| Skill | File | When to load |
|-------|------|-------------|
| Implementing RFCs | implementing-rfcs.md | Turning SPECIFICATION.md RFCs into Rust code |
| Testing | testing.md | Writing tests, debugging test failures, coverage |
| Burn Backend | burn-backend.md | Tensor operations, backend selection, burn API |
| Debugging | debugging.md | Build failures, test failures, numerical issues |
| Compat & Loading | compat-loading.md | safetensors, ONNX, checkpoint loading, key remapping |
| CI & Verification | ci-verification.md | CI pipeline, quality gates, release readiness |
</skills>

<current_state>

- All verification passes locally: build, 356 tests, clippy, fmt, docs.
- Strict masked-encoder paths: `IJepa::forward_step_strict` (image), `VJepa::forward_step_strict` (video).
- Differential parity: 3 checked-in strict I-JEPA image fixtures run in CI.
- Safetensors checkpoint loading functional. ONNX metadata + initializer loading work. Tract-based runtime exists.
- CLI: 6 subcommands (`models`, `inspect`, `checkpoint`, `train`, `encode`, `tui`).
- TUI: 5 tabs (Dashboard, Models, Training, Checkpoint, About).
- Release-candidate rehearsal complete locally; external crates.io publish pending approval.
- Still missing: strict video parity proof, crates.io release.
- Active planning: `PRODUCTION_GAP.md`, `ROADMAP.md`, `WORK_PACKAGES.md`.

</current_state>

<memory>

<decisions>
2026-03 Strict masked paths alongside generic — Faithful JEPA requires pre-attention masking; generic trainer can't do this with opaque Encoder::Input — Rejected: modifying Encoder trait (too invasive)
2026-03 thiserror 2 for all error types — Typed errors over panics for caller-triggerable misuse — Rejected: anyhow (loses type info)
2026-03 ONNX scoped to metadata/initializers — Runtime execution via tract exists but not proven production-grade — Rejected: claiming full ONNX support prematurely
2026-03 No optimizer in jepa-train — Caller owns optimization per burn convention — Rejected: bundled optimizer (limits flexibility)
</decisions>

<lessons>
- Mask disjointness must be validated early — silent overlap causes subtle parity bugs.
- Representation::gather must preserve masks — this was a regression once, caught by tests.
- TransformerPredictor needs real flattened indices, not relative offsets — mismatch causes silent wrong results.
- Config .validate() catches dimension mismatches early — always call before use.
</lessons>

</memory>
