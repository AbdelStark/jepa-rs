---
name: testing
description: Testing strategy for jepa-rs across 4 layers — unit tests, BDD/Gherkin, differential testing against Python, and fuzz testing. Activate when writing any test, discussing test strategy, or debugging test failures.
prerequisites: cargo test must be runnable
---

# Testing

<purpose>
Ensure correctness of numerical ML code through layered testing.
Numerical code is especially error-prone — subtle bugs produce wrong results without crashing.
</purpose>

<context>
4 test layers defined in SPECIFICATION.md:

| Layer | Type         | Framework    | Location                      | Status          |
|-------|--------------|--------------|-------------------------------|-----------------|
| 1     | Unit (TDD)   | built-in     | #[cfg(test)] in source        | 88 tests passing |
| 2     | BDD          | (planned)    | specs/gherkin/                | 27 scenarios written, not wired |
| 3     | Differential | safetensors  | tests/differential/ (planned) | Not started   |
| 4     | Fuzz         | libfuzzer    | tests/fuzz/ (planned)         | Not started     |

Test backend: `type TestBackend = burn_ndarray::NdArray<f32>;`
Dev dependency: burn-ndarray (already in all crate Cargo.toml files)

Current test distribution:
- types.rs: 12 tests (gather, validation, construction)
- config.rs: 18 tests (validation, presets, builder, serialization)
- energy.rs: 18 tests (8 unit + 4 proptest families)
- masking.rs: 14 tests (9 unit + 5 proptest families)
- collapse.rs: 21 tests (VICReg + BarlowTwins)
- ema.rs: 27 tests (15 unit + 3 proptest families)
- encoder.rs: 1 test (trait validation)
- predictor.rs: 2 tests (trait validation)
- 6 doc tests (config, energy, masking, collapse, ema)
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
   - Invariants (energy >= 0, mask coverage, momentum bounds, etc.)
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
6. Run: `cargo test -p jepa-core` (must see all 88+ tests pass)

Writing differential tests (when Python references exist):

1. Generate test fixtures from Python: save inputs/outputs as .safetensors
2. Place in tests/fixtures/[rfc-name]/
3. Load in Rust test with safetensors crate
4. Compare with tolerance: assert!((rust_output - python_output).abs() < 1e-5)
</procedure>

<patterns>
<do>
  — Test invariants, not implementation details
  — Use exact values for deterministic operations, tolerances for floating-point
  — Test tensor shapes explicitly: `assert_eq!(result.dims(), [batch, seq, dim])`
  — Name tests descriptively: `test_[what]_[condition]_[expected]`
  — Use proptest for properties that should hold for any valid input
  — Study existing tests in energy.rs, collapse.rs, ema.rs for patterns
  — Add doc tests to public types/traits (see config.rs, energy.rs for examples)
</do>
<dont>
  — Don't test private functions directly — test through the public API
  — Don't use `assert_eq!` for floating-point — use tolerance-based comparison
  — Don't skip edge cases — zero-length, single-element, max-size inputs
  — Don't couple tests to specific tensor values that might change with implementation
  — Don't break existing tests when adding new ones
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

Doc test pattern (from config.rs):

```rust
/// # Example
/// ```
/// use jepa_core::config::JepaConfigBuilder;
/// let config = JepaConfigBuilder::new()
///     .embed_dim(768)
///     .num_heads(12)
///     .build()
///     .unwrap();
/// assert_eq!(config.embed_dim, 768);
/// ```
```
</examples>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| `cargo test` compiles nothing new | Test module behind `#[cfg(test)]` not triggered | Ensure `cargo test -p [crate]` targets the right crate |
| Floating-point assertion fails intermittently | Tolerance too tight | Widen to 1e-5 or 1e-4 for accumulated operations |
| proptest shrinking takes too long | Input space too large | Add `#[proptest_config(ProptestConfig::with_cases(100))]` |
| proptest regression file appears | Shrunk failure saved | Check proptest-regressions/ — fix the bug, don't delete the file |

</troubleshooting>

<references>
— crates/jepa-core/src/energy.rs: Proptest + unit test examples (18 tests)
— crates/jepa-core/src/collapse.rs: Complex numerical test examples (21 tests)
— crates/jepa-core/src/ema.rs: Convergence + schedule testing (27 tests)
— crates/jepa-core/src/config.rs: Builder validation + serialization tests (18 tests)
— specs/gherkin/features.feature: 27 BDD scenarios with test vectors
— SPECIFICATION.md: Test vectors embedded in RFC sections
</references>
