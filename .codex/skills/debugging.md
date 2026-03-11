---
name: debugging
description: Diagnosing and fixing build failures, test failures, and numerical issues in jepa-rs. Activate when encountering compile errors, test failures, panics, or unexpected numerical results.
prerequisites: none
---

# Debugging

<purpose>
Fix common issues in this Rust ML workspace.
Build errors are the most frequent issue due to the project's pre-alpha state.
</purpose>

<context>
Current known state:
- `cargo build` FAILS: lib.rs re-exports 7 types from placeholder modules
- Only crates/jepa-core/src/types.rs has real implementation
- All other modules are single-line doc comment placeholders
- Tests only exist in types.rs (6 passing when compiled in isolation)
</context>

<procedure>
Build failure triage:

1. Run `cargo build 2>&1` and read the FIRST error (not the last)
2. Classify the error:
   - E0432 (unresolved import): A re-exported type doesn't exist yet → implement it or remove the re-export
   - E0405 (not found in scope): Missing use/import → add the correct `use` statement
   - E0308 (type mismatch): Usually tensor rank or backend mismatch → check generics
   - E0599 (method not found): burn API changed or wrong tensor rank → check burn 0.16 docs
3. Fix the root cause, not the symptom
4. After fixing, run `cargo build` again — fix errors one at a time

Test failure triage:

1. Run `cargo test -p [crate] -- --nocapture` to see output
2. For assertion failures: compare expected vs actual values
3. For floating-point: check if tolerance is appropriate (1e-6 for single ops, 1e-4 for chains)
4. For panics: read the backtrace — `RUST_BACKTRACE=1 cargo test`

Numerical debugging:

1. Print intermediate tensor values: `println!("{:?}", tensor.to_data());`
2. Compare shapes at each step: `println!("shape: {:?}", tensor.dims());`
3. Check for NaN/Inf: `assert!(!value.is_nan() && !value.is_infinite());`
4. Compare against Python reference: use safetensors fixtures
</procedure>

<patterns>
<do>
  — Read the first compiler error, not the last — later errors are often cascading
  — Use `cargo check` for faster compile-error feedback (no codegen)
  — Use `cargo test -- [test_name]` to run a single test
  — Use `RUST_BACKTRACE=1` for runtime panics
</do>
<dont>
  — Don't suppress warnings with `#[allow(...)]` without understanding them
  — Don't `unwrap()` in library code to "fix" a compile error — propagate with `?`
  — Don't ignore clippy suggestions — they often prevent real bugs
</dont>
</patterns>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| 7 E0432 errors on `cargo build` | Placeholder modules don't define re-exported types | Implement types per RFC, or comment out re-exports in lib.rs |
| `cargo test` shows 0 tests | Build fails before test discovery | Fix build errors first |
| Test passes locally, fails in CI | RNG seed not fixed, or platform-specific float | Use deterministic seed, widen tolerance |
| Infinite compile time | burn macro expansion | Expected for first build — burn is heavy. Use `cargo check` for iteration |

</troubleshooting>

<references>
— crates/jepa-core/src/lib.rs: Re-export declarations causing current build failure
— crates/jepa-core/src/types.rs: Working implementation to reference
</references>
