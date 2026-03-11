<identity>
jepa-rs: Production-grade Rust implementation of JEPA (Joint Embedding Predictive Architecture) for self-supervised learning.
Pre-alpha / specification phase. Most modules are placeholders awaiting implementation per SPECIFICATION.md RFCs.
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
| Benchmarks  | criterion    | 0.5      | Not yet wired up                         |
| RNG         | rand + chacha| 0.8/0.3  | Deterministic seeded RNG                 |

</stack>

<structure>
```
crates/
├── jepa-core/      # Core traits + types. ONLY crate with real code. [agent: create/modify]
│   └── src/
│       ├── lib.rs       # Module root, trait re-exports
│       ├── types.rs     # Representation, Energy, MaskSpec, InputShape (IMPLEMENTED)
│       ├── encoder.rs   # Encoder trait (PLACEHOLDER)
│       ├── predictor.rs # Predictor trait (PLACEHOLDER)
│       ├── energy.rs    # EnergyFn trait (PLACEHOLDER)
│       ├── masking.rs   # MaskingStrategy trait (PLACEHOLDER)
│       ├── collapse.rs  # CollapseRegularizer trait (PLACEHOLDER)
│       ├── ema.rs       # EMA struct (PLACEHOLDER)
│       └── config.rs    # JepaConfig (PLACEHOLDER)
├── jepa-vision/    # Vision-specific (ViT, I-JEPA, V-JEPA) [agent: create/modify]
├── jepa-world/     # World model / action-conditioned [agent: create/modify]
├── jepa-train/     # Training loop utilities [agent: create/modify]
└── jepa-compat/    # PyTorch checkpoint loading [agent: create/modify]
specs/
└── gherkin/
    └── features.feature  # BDD scenarios [agent: modify with care]
SPECIFICATION.md          # RFC archive (10 RFCs) — the implementation bible [agent: READ ONLY]
```
</structure>

<commands>

| Task            | Command                           | Notes                                       |
|-----------------|-----------------------------------|---------------------------------------------|
| Build           | `cargo build`                     | Currently fails — see known issues          |
| Build (release) | `cargo build --release`           | Same failure state                          |
| Test            | `cargo test`                      | Only jepa-core/types.rs has tests (6 pass)  |
| Test (single)   | `cargo test -p jepa-core`         | Target specific crate                       |
| Clippy          | `cargo clippy --all-targets`      | No clippy.toml — uses defaults              |
| Format          | `cargo fmt`                       | No rustfmt.toml — uses defaults             |
| Format (check)  | `cargo fmt -- --check`            | CI-style check                              |
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
    — Validate inputs at construction (see MaskSpec::validate pattern)
    — Write tests BEFORE implementation (TDD per SPECIFICATION.md)
    — Reference the specific RFC in SPECIFICATION.md when implementing a module
    — Use deterministic seeded RNG (rand_chacha) for reproducibility
    — Keep trait definitions minimal — one concern per trait
  </do>
  <dont>
    — Don't use raw `Tensor<B, N>` in public APIs — wrap in semantic types
    — Don't hardcode backend — always use `B: Backend` generic
    — Don't skip validation — mask indices, tensor shapes, config values must be checked
    — Don't add Python/PyTorch patterns — use idiomatic Rust (Result, iterators, ownership)
    — Don't implement beyond what the RFC specifies without discussion
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

<fix_build>
  Current state: lib.rs re-exports items from placeholder modules that don't define them.
  To fix: implement the trait/struct in each placeholder module, OR comment out the re-export.
  The placeholder modules contain only doc comments — no actual type definitions.
</fix_build>
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
| `cargo build` fails with E0432 errors    | lib.rs re-exports types from placeholder modules       | Implement the types, or comment out re-exports   |
| 7 unresolved import errors               | JepaConfig, Encoder, EnergyFn, etc. not yet defined   | Implement per SPECIFICATION.md RFCs              |
| Tests only run for types.rs              | Other modules are placeholders with no test targets    | Expected — implement modules to add tests        |

</known_issues>

<recovery_patterns>
  1. Read the full error — Rust errors are precise and usually contain the fix
  2. Check SPECIFICATION.md for the intended design of the failing component
  3. `cargo clean && cargo build` if incremental compilation is stale
  4. For burn API questions, check burn 0.16 docs
</recovery_patterns>
</troubleshooting>

<skills>
Modular skills in .codex/skills/ (symlinked at .claude/skills/ and .agents/skills/):
— _index.md: Skill registry and gap analysis
— implementing-rfcs.md: Step-by-step guide for turning SPECIFICATION.md RFCs into code
— testing.md: Test strategy across 4 layers (unit, BDD, differential, fuzz)
— burn-backend.md: Working with the burn ML framework in this project
— debugging.md: Diagnosing build failures, test failures, and numerical issues
</skills>

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
</lessons_learned>
</memory>
