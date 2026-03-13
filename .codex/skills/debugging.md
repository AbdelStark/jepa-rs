---
name: debugging
description: Diagnosing and fixing build failures, test failures, and numerical issues in jepa-rs. Activate when encountering compile errors, test failures, panics, or unexpected numerical results.
prerequisites: none
---

# Debugging

<purpose>
Fix common issues in this Rust ML workspace.
The workspace builds cleanly and 356 tests pass across 6 crates + 1 binary. Issues typically arise when adding new code.
</purpose>

<context>
Current known state:
- `cargo build` succeeds — all 6 library crates + binary compile
- `cargo test --workspace` runs 356 tests (355 pass, 1 ignored)
- jepa-core: ~99 tests (types, config, energy, masking, collapse, ema, encoder, predictor)
- jepa-vision: ~72 tests (ViT, patch, rope, image, video) + integration tests
- jepa-world: ~47 tests (action, planner, hierarchy, memory)
- jepa-train: ~31 tests (trainer, schedule, checkpoint)
- jepa-compat: ~16 tests (safetensors, onnx, keymap, registry)
- jepa (binary): ~11 tests (CLI commands)
- proptest regression files exist in crates/jepa-core/proptest-regressions/
- CI enforces: check, test, clippy, fmt, doc, coverage (80%), bench-smoke, parity, package-smoke, fuzz, audit
</context>

<procedure>
Build failure triage:

1. Run `cargo check 2>&1` (faster than full build) and read the FIRST error
2. Classify the error:
   - E0432 (unresolved import): Missing `pub use` or wrong path → fix import
   - E0405 (not found in scope): Missing `use` statement → add correct import
   - E0308 (type mismatch): Usually tensor rank or backend mismatch → check generics
   - E0599 (method not found): burn API issue or wrong tensor rank → check burn 0.20.1 docs
3. Fix the root cause, not the symptom
4. After fixing, run `cargo check` again — fix errors one at a time

Test failure triage:

1. Run `cargo test -p [crate] -- --nocapture` to see output
2. For assertion failures: compare expected vs actual values
3. For floating-point: check if tolerance is appropriate (1e-6 for single ops, 1e-4 for chains)
4. For panics: read the backtrace — `RUST_BACKTRACE=1 cargo test`
5. For proptest failures: check proptest-regressions/ for the shrunk case

Numerical debugging:

1. Print intermediate tensor values: `println!("{:?}", tensor.to_data());`
2. Compare shapes at each step: `println!("shape: {:?}", tensor.dims());`
3. Check for NaN/Inf: `assert!(!value.is_nan() && !value.is_infinite());`
4. Compare against Python reference: use safetensors fixtures in tests/fixtures/
</procedure>

<patterns>
<do>
  — Read the first compiler error, not the last — later errors are often cascading
  — Use `cargo check` for faster compile-error feedback (no codegen)
  — Use `cargo test -p [crate] -- [test_name]` to run a single test
  — Use `RUST_BACKTRACE=1` for runtime panics
  — Run `cargo test -p jepa-core` to verify existing tests still pass
</do>
<dont>
  — Don't suppress warnings with `#[allow(...)]` without understanding them
  — Don't `unwrap()` in library code to "fix" a compile error — propagate with `?`
  — Don't ignore clippy suggestions — they often prevent real bugs
  — Don't delete proptest-regressions/ files — they capture known edge cases
</dont>
</patterns>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| New E0432 after adding type to outer crate | Missing `pub use` or wrong import path | Check the crate's lib.rs re-exports |
| `cargo test` shows fewer tests than expected | Test filter active or compile error in one module | Run without filter: `cargo test -p [crate]` |
| Test passes locally, fails in CI | RNG seed not fixed, or platform-specific float | Use deterministic seed, widen tolerance |
| Infinite compile time on change | burn macro re-expansion | Expected for first build — subsequent builds are incremental |
| clippy warns about unnecessary clone | burn tensor ops consume the tensor | Check if `.clone()` is actually needed before the consuming op |

</troubleshooting>

<references>
— crates/jepa-core/src/lib.rs: Re-export declarations
— crates/jepa-core/proptest-regressions/: Saved proptest failure cases
— crates/jepa-core/src/types.rs: Foundation types all modules depend on
— .github/workflows/ci.yml: CI pipeline with 11 verification jobs
</references>
