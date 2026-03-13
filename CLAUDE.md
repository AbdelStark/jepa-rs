<identity>
jepa-rs is a Rust Cargo workspace for JEPA primitives, strict image and video reference paths, checkpoint and ONNX interoperability, and a CLI/TUI built on burn.
</identity>

<stack>

| Layer | Technology | Version | Notes |
|-------|------------|---------|-------|
| Runtime | Rust toolchain | stable, MSRV 1.85 | CI uses stable; nightly only for `fuzz/` |
| Language | Rust | 2021 edition | `rust-version = 1.85` in workspace manifests |
| Workspace | Cargo workspace | resolver 2 | 6 member crates plus nested fuzz workspace |
| ML framework | burn | 0.20.1 | `autodiff` enabled workspace-wide |
| CPU backend | burn-ndarray | 0.20.1 | Default test and demo backend |
| GPU/WebGPU backend | burn-wgpu | 0.20.1 | Present as workspace dependency |
| Serialization | serde, serde_json | 1.x | Configs and metadata |
| Errors | thiserror | 2.x | Library crates |
| Checkpoint format | safetensors | 0.7.0 | `jepa-compat` |
| ONNX runtime | tract-onnx | 0.23.0-dev.2 | `jepa-compat` runtime path |
| CLI | clap | 4.x | `crates/jepa` |
| TUI | ratatui, crossterm | 0.29, 0.28 | `crates/jepa` |
| Testing | cargo test, proptest | 1.x | Unit, integration, doc, property tests |
| Benchmarks | criterion | 0.8.2 | Per-crate benches |
| Fuzzing | cargo-fuzz, libfuzzer-sys | nightly, 0.4.12 | Separate `fuzz/` workspace |

</stack>

<structure>
```text
crates/
├── jepa-core/      # Shared contracts, semantic tensor wrappers, masking, energy, config [agent: gated for public API files]
├── jepa-vision/    # ViT, IJepa, VJepa, strict image/video reference paths [agent: create/modify]
├── jepa-train/     # Generic training orchestration and schedules [agent: create/modify]
├── jepa-world/     # Planning, memory, hierarchical world-model helpers [agent: create/modify]
├── jepa-compat/    # safetensors, key mapping, ONNX metadata/runtime [agent: create/modify]
└── jepa/           # CLI binary, demos, and ratatui dashboard [agent: create/modify]
.github/workflows/  # CI gates and release smoke checks [agent: gated]
docs/               # Project docs [agent: create/modify]
docs/agentic/       # Living agent memory: decisions and lessons learned [agent: create/modify]
fuzz/               # Separate cargo-fuzz workspace and fuzz targets [agent: modify with care]
scripts/            # Parity and export scripts [agent: gated]
specs/differential/ # Strict I-JEPA parity fixtures and fixture tooling [agent: gated]
target/             # Generated build outputs and demo artifacts [agent: never edit]
README.md           # Public project overview [agent: create/modify]
CONTRIBUTING.md     # Dev workflow and commit convention [agent: create/modify]
CHANGELOG.md        # Release history [agent: create/modify]
Cargo.toml          # Workspace membership and shared dependency versions [agent: gated]
Cargo.lock          # Locked dependency graph [agent: gated]
```

Module boundaries:
- `jepa-core` is the contract layer. All library crates depend on it.
- `jepa-vision`, `jepa-train`, `jepa-world`, and `jepa-compat` are library layers on top of `jepa-core`.
- `crates/jepa` is the only binary crate and depends on all other workspace crates.
- `fuzz/` is not a normal workspace member. Treat it as a separate nightly-only validation surface.
</structure>

<commands>

| Task | Command | Notes |
|------|---------|-------|
| Workspace check | `cargo check --workspace --all-targets` | Fastest whole-repo compile gate |
| Workspace tests | `cargo test --workspace` | Runs unit, integration, and doc tests |
| Core-only tests | `cargo test -p jepa-core` | Use before wider workspace runs |
| Vision tests | `cargo test -p jepa-vision` | Includes strict-path unit tests |
| Compat tests | `cargo test -p jepa-compat` | safetensors, ONNX, keymap coverage |
| CLI/TUI tests | `cargo test -p jepa` | Clap parsing and demo helpers |
| Clippy | `cargo clippy --workspace --all-targets -- -D warnings` | Warnings are errors in CI |
| Format check | `cargo fmt -- --check` | Use `cargo fmt` only to apply formatting |
| Strict parity | `scripts/run_parity_suite.sh` | Runs all bundled strict image fixtures |

CI also runs additional gates from `.github/workflows/ci.yml`: `cargo doc --no-deps --all-features`, `cargo llvm-cov --workspace --all-features --fail-under-lines 80 --summary-only`, `cargo bench --workspace --no-run`, crate packaging smoke checks, nightly fuzz smoke, and `rustsec` audit.
</commands>

<conventions>
  <code_style>
    Files and modules use `snake_case`. Types use `PascalCase`. Constants use `SCREAMING_SNAKE_CASE`.
    Keep public tensor-bearing APIs generic over `B: Backend`.
    Prefer semantic wrappers such as `Representation<B>`, `Energy<B>`, and `Action<B>` over raw public `Tensor` types when a wrapper already exists.
    Library crates use typed `thiserror` enums. CLI and TUI code in `crates/jepa` uses `anyhow::Result` plus `.context(...)`.
    Tests live in `#[cfg(test)] mod tests` at the bottom of the owning file and use `burn_ndarray::NdArray<f32>` on CPU unless a test proves otherwise.
    Let `rustfmt` own import ordering. Prefer `use crate::...` inside a crate and explicit crate names across crates.
  </code_style>

  <patterns>
    <do>
      - Call `.validate()` on configs before trusting user-provided or synthesized values.
      - Preserve mask metadata through gather and slicing operations; `Representation::gather` is the reference behavior.
      - Be explicit about strict versus approximate training semantics in comments, docs, and code review notes.
      - Add regression tests for behavioral fixes, especially around masking, target positions, and checkpoint loading.
      - Keep demos and smoke tests small, deterministic, and CPU-friendly.
    </do>
    <dont>
      - Do not use `unwrap()` or `expect()` in library code. Return typed errors instead.
      - Do not treat `jepa_train::JepaComponents::forward_step` as the semantic reference path for strict masking.
      - Do not weaken parity tolerances or delete regression artifacts to make tests pass.
      - Do not add dependencies, features, or manifest changes without approval.
      - Do not update CLI flags without updating clap tests in `crates/jepa/src/cli.rs`.
    </dont>
  </patterns>

  <commit_conventions>
    Format commits as `type(scope): description`.
    Allowed scopes in `CONTRIBUTING.md`: `core`, `vision`, `world`, `train`, `compat`, `cli`, `specs`.
  </commit_conventions>
</conventions>

<workflows>
  <new_feature>
    1. Route the task to the smallest owning crate.
    2. If the change requires `Cargo.toml`, `Cargo.lock`, CI files, scripts, parity fixtures, or `jepa-core` public contract files, stop and ask for approval.
    3. Add or update the narrowest useful tests first in the owning crate.
    4. Implement the feature using existing crate-local patterns before introducing new helpers.
    5. Run the owning crate tests, then `cargo check --workspace --all-targets`.
    6. Run `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo fmt -- --check`.
    7. If strict image semantics changed, run `scripts/run_parity_suite.sh`.
  </new_feature>

  <bug_fix>
    1. Reproduce the issue with a failing unit, integration, or parity test.
    2. Fix the smallest layer that can own the behavior.
    3. Re-run the targeted crate test command until green.
    4. Expand to workspace-wide verification.
    5. Check whether the fix changed public behavior and document it in `README.md` or `CHANGELOG.md` if needed.
  </bug_fix>

  <checkpoint_or_onnx_change>
    1. Decide whether the task is safetensors loading, key remapping, ONNX metadata, ONNX runtime execution, or CLI encode behavior.
    2. Keep format-specific logic inside `jepa-compat`.
    3. Prefer typed errors that distinguish missing files, invalid formats, and shape mismatches.
    4. Run `cargo test -p jepa-compat` and `cargo test -p jepa` when CLI encode behavior changed.
    5. If ONNX runtime or export assumptions changed, inspect `scripts/export_ijepa_onnx.py` and note any external Python requirements.
  </checkpoint_or_onnx_change>
</workflows>

<boundaries>
  <zone_map>

| Zone | Paths | Rule |
|------|-------|------|
| Autonomous | `crates/jepa-vision/src/`, `crates/jepa-train/src/`, `crates/jepa-world/src/`, `crates/jepa-compat/src/`, `crates/jepa/src/`, `crates/*/examples/`, `crates/*/benches/`, `docs/`, `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `fuzz/fuzz_targets/`, `.codex/skills/` | Agent may create and modify freely, then run local verification |
| Gated | `Cargo.toml`, `Cargo.lock`, `crates/*/Cargo.toml`, `fuzz/Cargo.toml`, `.github/workflows/`, `scripts/`, `specs/differential/`, `crates/jepa-core/src/lib.rs`, `crates/jepa-core/src/encoder.rs`, `crates/jepa-core/src/predictor.rs`, `crates/jepa-core/src/energy.rs`, `crates/jepa-core/src/masking.rs`, `crates/jepa-core/src/collapse.rs`, `crates/jepa-core/src/types.rs` | Read freely. Ask before editing |
| Forbidden | `.env`, `.env.*`, `.git/`, `target/`, `fuzz/target/` | Never read or modify secrets or generated internals |

  </zone_map>

  <safety_checks>
    Before any destructive operation, fixture rewrite, or overwrite of a gated path:
    1. State the exact file or command.
    2. State the failure mode.
    3. Wait for confirmation.
  </safety_checks>
</boundaries>

<troubleshooting>
  <known_issues>

| Symptom | Cause | Fix |
|---------|-------|-----|
| `input shape mismatch: expected ... got ...` | ONNX model has static or differently-bound input dims | Use `OnnxEncoder::from_path_with_input_shape(...)` or align the caller shape |
| `WarmupExceedsTotal` or invalid train CLI config | `warmup` is larger than `steps`, or another config invariant failed | Fix the CLI args and re-run config validation |
| `test_ijepa_strict_fixture_parity` appears ignored | The parity test is intentionally gated behind the fixture runner | Run `scripts/run_parity_suite.sh` from repo root |
| `failed to inject weights from ...` | safetensors checkpoint shape or key mapping does not match the chosen preset | Use the matching `VitConfig` preset and `ijepa_vit_keymap()` path |
| `expected Tensor<_, 3> found Tensor<_, 2>` | A `reshape`, `sum`, or `squeeze` changed rank unexpectedly | Inspect intermediate dims and restore `[batch, seq, embed]` shape |
| `Blocking waiting for file lock on build directory` | Multiple cargo jobs are sharing the same checkout | Serialize cargo commands or wait for the competing job to finish |

  </known_issues>

  <recovery_patterns>
    1. Read the first compiler or runtime error before looking at cascaded failures.
    2. Run the narrowest crate command that can reproduce the problem.
    3. If the change touched strict image behavior, run `scripts/run_parity_suite.sh`.
    4. Check `git status --short` for overlapping in-flight changes before assuming the repo is clean.
    5. If you suspect a shared contract regression, review `crates/jepa-core/src/lib.rs` and the related public type file.
  </recovery_patterns>
</troubleshooting>

<environment>
  Harness: terminal agents that read `CLAUDE.md` and `.codex/skills/*.md`
  File system scope: the full repository root, including the nested `fuzz/` workspace
  Network access: optional [verify]. Required for external model downloads and Python export workflows
  Tool access: `git`, `cargo`, `rustfmt`, `clippy`, `python3`, and shell utilities
  Human interaction model: synchronous chat. Ask before gated edits and destructive operations
</environment>

<skills>
  Repo-local skills live in `.codex/skills/` with symlinks at `.claude/skills/` and `.agents/skills/`.
  Load only the file that matches the active domain.

  Available skills:
  - `workspace-development.md`: route work to the correct crate and keep shared-contract changes contained
  - `strict-vision-models.md`: strict image and video semantics, burn tensor shape discipline, parity-sensitive changes
  - `testing-and-parity.md`: unit, integration, property, fuzz, and strict parity workflows
  - `checkpoint-and-onnx.md`: safetensors, key maps, ONNX metadata, runtime, and export assumptions
  - `cli-and-demos.md`: clap surfaces, command reporters, demo flows, and TUI event handling
</skills>

<memory>
  <project_decisions>
    Source of truth: `docs/agentic/project-decisions.md`
    Update it when a task introduces or confirms a durable architectural or workflow decision.
  </project_decisions>

  <lessons_learned>
    Source of truth: `docs/agentic/lessons-learned.md`
    Append short, reusable entries when a failure mode, debugging shortcut, or verification rule proves useful more than once.
  </lessons_learned>
</memory>
