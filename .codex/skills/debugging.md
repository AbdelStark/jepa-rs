---
name: debugging
description: Diagnosing and fixing build failures, test failures, and numerical issues in jepa-rs. Activate when encountering compile errors, test failures, panics, or unexpected numerical results.
prerequisites: none
---

# Debugging

<purpose>
Fix common issues in this Rust ML workspace.
The workspace builds cleanly and 94 tests pass (88 unit + 6 doc). Issues typically arise when adding new code.
</purpose>

<context>
Current known state:
- `cargo build` succeeds — all 5 crates compile
- `cargo test -p jepa-core` runs 88 unit tests + 6 doc tests (all pass)
- jepa-core has 8 implemented modules; outer crates are stubs
- proptest regression files exist in crates/jepa-core/proptest-regressions/
</context>

<procedure>
Build failure triage:

1. Run `cargo check 2>&1` (faster than full build) and read the FIRST error
2. Classify the error:
   - E0432 (unresolved import): A re-exported type doesn't exist yet → implement it or fix the import path
   - E0405 (not found in scope): Missing use/import → add the correct `use` statement
   - E0308 (type mismatch): Usually tensor rank or backend mismatch → check generics
   - E0599 (method not found): burn API issue or wrong tensor rank → check burn 0.16 docs
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
4. Compare against Python reference: use safetensors fixtures
</procedure>

<patterns>
<do>
  — Read the first compiler error, not the last — later errors are often cascading
  — Use `cargo check` for faster compile-error feedback (no codegen)
  — Use `cargo test -- [test_name]` to run a single test
  — Use `RUST_BACKTRACE=1` for runtime panics
  — Run `cargo test -p jepa-core` to verify you haven't broken existing tests
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
| `cargo test` shows fewer than 88 tests | Test filter active or compile error in one module | Run without filter: `cargo test -p jepa-core` |
| Test passes locally, fails in CI | RNG seed not fixed, or platform-specific float | Use deterministic seed, widen tolerance |
| Infinite compile time on change | burn macro re-expansion | Expected for first build — subsequent builds are incremental |
| clippy warns about unnecessary clone | burn tensor ops consume the tensor | Check if `.clone()` is actually needed before the consuming op |

</troubleshooting>

<references>
— crates/jepa-core/src/lib.rs: Re-export declarations
— crates/jepa-core/proptest-regressions/: Saved proptest failure cases
— crates/jepa-core/src/types.rs: Foundation types all modules depend on
</references>
