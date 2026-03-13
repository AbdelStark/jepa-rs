---
name: testing
description: Testing strategy for jepa-rs across 4 layers — unit tests, BDD/Gherkin, differential parity against Python, and fuzz testing. Activate when writing any test, discussing test strategy, or debugging test failures.
prerequisites: cargo test must be runnable
---

# Testing

<purpose>
Ensure correctness of numerical ML code through layered testing.
Numerical code is especially error-prone — subtle bugs produce wrong results without crashing.
</purpose>

<context>
4 test layers:

| Layer | Type         | Framework    | Location                        | Status                                    |
|-------|--------------|--------------|-------------------------------- |-------------------------------------------|
| 1     | Unit (TDD)   | built-in     | `#[cfg(test)]` in source        | 356 tests passing across workspace        |
| 2     | BDD          | (planned)    | specs/gherkin/                  | 27 scenarios written, not wired           |
| 3     | Differential | safetensors  | jepa-vision/tests/integration.rs| 3 I-JEPA image fixtures in CI             |
| 4     | Fuzz         | libfuzzer    | fuzz/                           | masking, gather, energy, checkpoint_parsing|

Test backend: `type TestBackend = burn_ndarray::NdArray<f32>;`
Dev dependency: burn-ndarray 0.20.1 (already in all crate Cargo.toml files)

CI gates: 80% line coverage floor via `cargo llvm-cov`, fuzz runs 256 iterations per target.

Test distribution by crate:
- jepa-core: ~99 tests (types, config, energy, masking, collapse, ema, encoder, predictor)
- jepa-vision: ~72 tests + integration parity tests
- jepa-world: ~47 tests
- jepa-train: ~31 tests
- jepa-compat: ~16 tests
- jepa (CLI): ~11 tests
</context>

<procedure>
Writing unit tests:

1. Create `#[cfg(test)] mod tests` at the bottom of the source file
2. Add `use super::*;` and any test-specific imports
3. Define `type TestBackend = burn_ndarray::NdArray<f32>;`
4. Write test functions covering:
   - Happy path with known expected values
   - Edge cases (empty, zero, single-element)
   - Error cases (invalid input → correct error variant)
   - Invariants (energy >= 0, mask coverage, momentum bounds)
5. For numerical properties, use proptest:
   ```rust
   use proptest::prelude::*;
   proptest! {
       #[test]
       fn energy_is_non_negative(x in 0.0f32..100.0) {
           // property assertion
       }
   }
   ```
6. Run: `cargo test -p [crate]` — all tests must pass

Writing differential tests (parity against Python reference):

1. Generate test fixtures from Python: save inputs/outputs as .safetensors
2. Place in tests/fixtures/[name]/
3. Load in Rust integration test with safetensors crate
4. Compare with tolerance: `assert!((rust_output - python_output).abs() < 1e-5)`
5. See `jepa-vision/tests/integration.rs` for the established pattern (ParityFixture)
</procedure>

<patterns>
<do>
  — Test invariants, not implementation details
  — Use exact values for deterministic operations, tolerances for floating-point
  — Test tensor shapes explicitly: `assert_eq!(result.dims(), [batch, seq, dim])`
  — Name tests descriptively: `test_[what]_[condition]_[expected]`
  — Use proptest for properties that should hold for any valid input
  — Add doc tests to public types/traits (see config.rs, energy.rs for examples)
  — Always verify mask preservation through gather operations
</do>
<dont>
  — Don't test private functions directly — test through the public API
  — Don't use `assert_eq!` for floating-point — use tolerance-based comparison
  — Don't skip edge cases — zero-length, single-element, max-size inputs
  — Don't couple tests to specific tensor values that might change with implementation
  — Don't break existing tests when adding new ones
  — Don't delete proptest-regressions/ files — they capture known edge cases
</dont>
</patterns>

<examples>
Proptest pattern (from energy.rs):

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn l2_energy_non_negative(
        a in prop::collection::vec(-10.0f32..10.0, 4..=4),
        b in prop::collection::vec(-10.0f32..10.0, 4..=4),
    ) {
        let device = burn_ndarray::NdArrayDevice::Cpu;
        // ... create representations from a, b ...
        let energy = L2Energy.compute(&rep_a, &rep_b);
        let val: f32 = energy.value.into_scalar();
        prop_assert!(val >= 0.0, "L2 energy must be non-negative, got {}", val);
    }
}
```

Differential parity pattern (from jepa-vision/tests/integration.rs):

```rust
struct ParityFixture { /* metadata + config */ }

#[test]
fn test_strict_image_parity() {
    let fixture = ParityFixture::load("tests/fixtures/ijepa_strict/");
    let mask = fixed_image_mask();
    mask.validate().unwrap();
    // Run strict forward, compare against fixture output with tolerance
}
```
</examples>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| `cargo test` compiles nothing new | Test module behind `#[cfg(test)]` not triggered | Ensure `cargo test -p [crate]` targets the right crate |
| Floating-point assertion fails intermittently | Tolerance too tight | Widen to 1e-5 or 1e-4 for accumulated operations |
| proptest shrinking takes too long | Input space too large | Add `#[proptest_config(ProptestConfig::with_cases(100))]` |
| proptest regression file appears | Shrunk failure saved | Check proptest-regressions/ — fix the bug, don't delete the file |
| Parity test fails in CI | Platform float differences | Use wider tolerance or deterministic seed |

</troubleshooting>

<references>
— crates/jepa-core/src/energy.rs: Proptest + unit test examples
— crates/jepa-core/src/collapse.rs: Complex numerical test examples
— crates/jepa-core/src/ema.rs: Convergence + schedule testing
— crates/jepa-vision/tests/integration.rs: Differential parity fixture pattern
— specs/gherkin/features.feature: 27 BDD scenarios with test vectors
— fuzz/: Fuzz targets (masking, gather, energy, checkpoint_parsing)
</references>
