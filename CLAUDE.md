<identity>
jepa-rs: Production-grade Rust implementation of JEPA (Joint Embedding Predictive Architecture) for self-supervised learning.
All 5 workspace crates are fully implemented with 245 unit/integration tests + 16 doc tests passing.
All 10 RFCs from SPECIFICATION.md are implemented across the workspace.
</identity>

<stack>

| Layer       | Technology   | Version  | Notes                                    |
|-------------|--------------|----------|------------------------------------------|
| Language    | Rust         | 2021 ed. | rustc 1.93+                              |
| ML backend  | burn         | 0.16     | Backend-agnostic: ndarray (CPU), wgpu (GPU) |
| Serializer  | serde        | 1        | With derive feature                      |
| Weights I/O | safetensors  | 0.4      | PyTorch/HuggingFace checkpoint loading   |
| Errors      | thiserror    | 2        | Derive macro for error enums             |
| Testing     | proptest     | 1        | Property-based testing                   |
| Benchmarks  | criterion    | 0.5      | All crates: core, vision, train, world     |
| RNG         | rand + chacha| 0.8/0.3  | Deterministic seeded RNG                 |

</stack>

<structure>
```
crates/
├── jepa-core/      # Core traits + types — primary implementation crate [agent: create/modify]
│   └── src/
│       ├── lib.rs       # Module root, trait re-exports (all resolve cleanly)
│       ├── types.rs     # Representation, Energy, MaskSpec, InputShape (IMPLEMENTED, 12 tests)
│       ├── config.rs    # JepaConfig, JepaConfigBuilder, presets (IMPLEMENTED, 18 tests)
│       ├── encoder.rs   # Encoder trait + IdentityEncoder test helper (IMPLEMENTED, 1 test)
│       ├── predictor.rs # Predictor trait + ZeroPredictor test helper (IMPLEMENTED, 2 tests)
│       ├── energy.rs    # EnergyFn, L2Energy, CosineEnergy, SmoothL1Energy (IMPLEMENTED, 18 tests)
│       ├── masking.rs   # MaskingStrategy, BlockMasking, SpatiotemporalMasking, MultiBlockMasking (IMPLEMENTED, 14 tests)
│       ├── collapse.rs  # CollapseRegularizer, VICReg, BarlowTwins (IMPLEMENTED, 21 tests)
│       └── ema.rs       # Ema, CosineMomentumSchedule (IMPLEMENTED, 27 tests)
├── jepa-vision/    # Vision-specific (ViT, I-JEPA, V-JEPA) — IMPLEMENTED [agent: create/modify]
│   └── src/
│       ├── vit.rs       # VitEncoder, TransformerBlock, MHSA, MLP (IMPLEMENTED, 10 tests + 3 proptests)
│       ├── patch.rs     # PatchEmbedding for image patchification (IMPLEMENTED, 5 tests + 2 proptests)
│       ├── rope.rs      # RotaryPositionEncoding2D for spatial awareness (IMPLEMENTED, 4 tests + 2 proptests)
│       ├── image.rs     # TransformerPredictor, IJepa model (IMPLEMENTED, 8 tests + 4 BDD + 2 proptests)
│       └── video.rs     # VitVideoEncoder, TubeletEmbedding, 3D RoPE, VJepa (IMPLEMENTED, 11 tests + 1 BDD)
├── jepa-world/     # World model / action-conditioned — IMPLEMENTED [agent: create/modify]
│   └── src/
│       ├── action.rs     # Action, ActionConditionedPredictor trait (IMPLEMENTED, 3 tests)
│       ├── planner.rs    # WorldModel, CEM planner, L2Cost (IMPLEMENTED, 7 tests + 2 proptests)
│       ├── hierarchy.rs  # HierarchicalJepa, JepaLevel (IMPLEMENTED, 3 tests)
│       └── memory.rs     # ShortTermMemory ring buffer (IMPLEMENTED, 10 tests + 3 proptests)
├── jepa-train/     # Training loop utilities — IMPLEMENTED [agent: create/modify]
│   └── src/
│       ├── trainer.rs    # JepaComponents, forward_step orchestration (IMPLEMENTED, 5 tests)
│       ├── schedule.rs   # WarmupCosineSchedule, LrSchedule trait (IMPLEMENTED, 6 tests + 3 proptests)
│       ├── checkpoint.rs # CheckpointMeta serialization (IMPLEMENTED, 3 tests)
│       └── step.rs       # TrainConfig, TrainMetrics, TrainStepOutput (IMPLEMENTED, 7 tests)
└── jepa-compat/    # PyTorch checkpoint loading — IMPLEMENTED [agent: create/modify]
    └── src/
        ├── safetensors.rs # Load/convert safetensors checkpoints (IMPLEMENTED, 12 tests)
        ├── keymap.rs      # I-JEPA/V-JEPA key mapping patterns (IMPLEMENTED, 11 tests + 6 proptests)
        └── onnx.rs        # ONNX model info types (API-complete, runtime stub, 8 tests)
specs/
└── gherkin/
    └── features.feature  # 27 BDD scenarios across 6 features [agent: modify with care]
SPECIFICATION.md          # RFC archive (10 RFCs, 1105 lines) — the implementation bible [agent: READ ONLY]
```
</structure>

<commands>

| Task            | Command                           | Notes                                       |
|-----------------|-----------------------------------|---------------------------------------------|
| Build           | `cargo build`                     | Succeeds — all workspace crates compile     |
| Build (release) | `cargo build --release`           | Succeeds                                    |
| Test            | `cargo test`                      | 245 unit/integration + 16 doc tests pass    |
| Test (single)   | `cargo test -p jepa-core`         | Target specific crate                       |
| Test (named)    | `cargo test -p jepa-core -- [name]` | Run a single test by name                 |
| Test (verbose)  | `cargo test -p jepa-core -- --nocapture` | Show println output                  |
| Clippy          | `cargo clippy --all-targets`      | No clippy.toml — uses defaults              |
| Format          | `cargo fmt`                       | No rustfmt.toml — uses defaults             |
| Format (check)  | `cargo fmt -- --check`            | CI-style check                              |
| Check           | `cargo check`                     | Fast compile check, no codegen              |
| Docs            | `cargo doc --open`                | Generate + open docs                        |

</commands>

<conventions>
<code_style>
  Naming: snake_case for functions/variables, PascalCase for types/traits, SCREAMING_SNAKE for constants.
  Files: snake_case.rs — one module per file.
  Generics: All ML types are generic over `B: Backend` (burn's backend trait).
  Error handling: thiserror derive macros. Domain-specific error enums per module.
  Tensor shapes: Document in doc comments as `[dim1, dim2, ...]`.
  Testing: #[cfg(test)] mod tests in same file. Property tests with proptest where applicable.
</code_style>

<patterns>
  <do>
    — Make all tensor types generic over `B: Backend` for backend-agnostic code
    — Use `Representation<B>`, `Energy<B>` wrapper types instead of raw tensors
    — Validate inputs at construction (see MaskSpec::validate, JepaConfig::validate patterns)
    — Write tests BEFORE implementation (TDD per SPECIFICATION.md)
    — Reference the specific RFC in SPECIFICATION.md when implementing a module
    — Use deterministic seeded RNG (rand_chacha) for reproducibility
    — Keep trait definitions minimal — one concern per trait
    — Use `#[derive(Debug, Clone)]` on all public structs
    — Use `thiserror::Error` for error enums with descriptive messages
  </do>
  <dont>
    — Don't use raw `Tensor<B, N>` in public APIs — wrap in semantic types
    — Don't hardcode backend — always use `B: Backend` generic
    — Don't skip validation — mask indices, tensor shapes, config values must be checked
    — Don't add Python/PyTorch patterns — use idiomatic Rust (Result, iterators, ownership)
    — Don't implement beyond what the RFC specifies without discussion
    — Don't use `unwrap()` in library code — return Result or propagate with `?`
  </dont>
</patterns>

<commit_conventions>
  Format: type(scope): description
  Types: feat, fix, refactor, test, docs, chore
  Scopes: core, vision, world, train, compat, specs
  Example: feat(core): implement EnergyFn trait with L2 and cosine variants
</commit_conventions>
</conventions>

<workflows>
<implement_rfc>
  1. Read the target RFC section in SPECIFICATION.md
  2. Read specs/gherkin/features.feature for relevant BDD scenarios
  3. Write trait/struct definitions with doc comments citing the RFC
  4. Write unit tests covering the RFC's test vectors
  5. Implement until tests pass
  6. Run `cargo test -p [crate]` — all must pass
  7. Run `cargo clippy --all-targets` — zero warnings
  8. Run `cargo fmt -- --check` — must pass
  9. Commit: feat(scope): implement [RFC component]
</implement_rfc>

<add_tests>
  1. Identify the invariant or behavior to test
  2. For deterministic properties: standard #[test]
  3. For statistical/range properties: proptest with proptest!{} macro
  4. Place tests in #[cfg(test)] mod tests at bottom of source file
  5. Use burn-ndarray backend for test execution: `type TestBackend = burn_ndarray::NdArray<f32>;`
</add_tests>

<next_implementation_targets>
  All RFCs are implemented. Remaining work:
  1. ONNX runtime integration (jepa-compat/onnx.rs — needs `ort` crate dependency)
  2. Distributed training support (out of scope for v0.1)
  3. Differential testing against Python reference implementations
  4. Fuzz testing targets for masking and energy functions
  5. Additional benchmarks for jepa-train and jepa-world
</next_implementation_targets>
</workflows>

<boundaries>
<forbidden>
  DO NOT modify:
  — SPECIFICATION.md (the RFC source of truth — changes need explicit approval)
  — LICENSE
</forbidden>

<gated>
  Modify ONLY with explicit approval:
  — Cargo.toml (workspace root — dependency changes)
  — crates/*/Cargo.toml (crate dependency changes)
  — Any public trait signature already implemented and tested
</gated>

<safety_checks>
  Before deleting any file or removing any public API:
  1. Check for dependents with `cargo doc` or grep for usage
  2. State what you're removing and why
  3. Wait for confirmation
</safety_checks>
</boundaries>

<troubleshooting>
<known_issues>

| Symptom                                  | Cause                                                  | Fix                                              |
|------------------------------------------|--------------------------------------------------------|--------------------------------------------------|
| Slow first build (~10s)                  | burn macro expansion is heavy                          | Expected — use `cargo check` for faster iteration |
| proptest shrinking takes long            | Input space too large                                  | Add `ProptestConfig::with_cases(100)` annotation |
| Float assertion fails intermittently     | Tolerance too tight for accumulated ops                | Widen to 1e-4 for chained operations             |
| ONNX test ignored                        | ort runtime crate not yet added as dependency          | Expected — add ort dependency when ready         |

</known_issues>

<recovery_patterns>
  1. Read the full error — Rust errors are precise and usually contain the fix
  2. Check SPECIFICATION.md for the intended design of the failing component
  3. `cargo clean && cargo build` if incremental compilation is stale
  4. For burn API questions, check burn 0.16 docs
  5. Use `cargo check` for fast compile feedback without codegen
</recovery_patterns>
</troubleshooting>

<skills>
Modular skills in .codex/skills/:
— _index.md: Skill registry and gap analysis
— implementing-rfcs.md: Step-by-step guide for turning SPECIFICATION.md RFCs into code
— testing.md: Test strategy across 4 layers (unit, BDD, differential, fuzz)
— burn-backend.md: Working with the burn ML framework in this project
— debugging.md: Diagnosing build failures, test failures, and numerical issues
</skills>

<ci>
  GitHub Actions CI (.github/workflows/ci.yml) runs on push/PR to main:
  — cargo check, cargo test, cargo clippy, cargo fmt --check, cargo doc
  All checks must pass before merge.
</ci>

<memory>
<project_decisions>
  2024: Use burn over tch-rs — Backend-agnostic, pure Rust, no C++ dependency — tch-rs rejected (FFI complexity)
  2024: Workspace with 5 crates — Separation of concerns: core traits, vision, world, training, compat — Monolithic crate rejected (coupling)
  2024: Semantic tensor wrappers — Representation<B>, Energy<B> prevent shape confusion — Raw tensors rejected (error-prone)
  2024: TDD + differential testing — Correctness verified against Python reference implementations — Unit-only rejected (insufficient for numerical code)
  2024: thiserror for errors — Lightweight, derive-based — anyhow rejected (library crate needs typed errors)
</project_decisions>

<lessons_learned>
  — Placeholder modules with re-exports cause compile errors. When stubbing, either define the type or don't re-export.
  — burn 0.16 uses `Backend` trait — all tensor ops must be generic over backend.
  — SPECIFICATION.md is the single source of truth for API design. Always read the RFC before implementing.
  — proptest is excellent for numerical invariants (energy >= 0, convergence, bounds, determinism).
  — JepaConfig builder pattern with validation-on-build prevents invalid configurations at construction.
  — VICReg and BarlowTwins implementations need careful attention to tensor broadcasting and covariance matrix computation.
</lessons_learned>
</memory>
