---
name: testing
description: Testing strategy for jepa-rs across 4 layers — unit tests, BDD/Gherkin, differential testing against Python, and fuzz testing. Activate when writing any test, discussing test strategy, or debugging test failures.
prerequisites: cargo test must be runnable (at least for implemented modules)
---

# Testing

<purpose>
Ensure correctness of numerical ML code through layered testing.
Numerical code is especially error-prone — subtle bugs produce wrong results without crashing.
</purpose>

<context>
4 test layers defined in SPECIFICATION.md:

| Layer | Type         | Framework    | Location                    | Status          |
|-------|--------------|--------------|-----------------------------|-----------------|
| 1     | Unit (TDD)   | built-in     | #[cfg(test)] in source      | 6 tests passing |
| 2     | BDD          | (planned)    | specs/gherkin/              | Scenarios written, not wired |
| 3     | Differential | safetensors  | tests/differential/ (planned) | Not started   |
| 4     | Fuzz         | libfuzzer    | tests/fuzz/ (planned)       | Not started     |

Test backend: `type TestBackend = burn_ndarray::NdArray<f32>;`
Dev dependency: burn-ndarray (already in jepa-core/Cargo.toml)
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
   - Invariants (energy >= 0, mask coverage = 1.0, etc.)
5. For numerical properties, use proptest:
   ```rust
   proptest! {
       #[test]
       fn energy_is_non_negative(x in 0.0f32..100.0) {
           // property assertion
       }
   }
   ```
6. Run: `cargo test -p jepa-core`

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
</do>
<dont>
  — Don't test private functions directly — test through the public API
  — Don't use `assert_eq!` for floating-point — use tolerance-based comparison
  — Don't skip edge cases — zero-length, single-element, max-size inputs
  — Don't couple tests to specific tensor values that might change with implementation
</dont>
</patterns>

<examples>
Unit test pattern (from types.rs):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_spec_validate_valid() {
        let mask = MaskSpec {
            context_indices: vec![0, 1, 2],
            target_indices: vec![3, 4, 5],
            total_tokens: 6,
        };
        assert!(mask.validate().is_ok());
    }
}
```

Floating-point tolerance pattern:

```rust
#[test]
fn test_energy_l2_known_value() {
    let device = burn_ndarray::NdArrayDevice::Cpu;
    // ... create tensors ...
    let energy_val: f32 = energy.value.into_scalar();
    assert!((energy_val - expected).abs() < 1e-6, "L2 energy mismatch");
}
```
</examples>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| `cargo test` compiles nothing new | Test module behind `#[cfg(test)]` not triggered | Ensure `cargo test -p [crate]` targets the right crate |
| Floating-point assertion fails intermittently | Tolerance too tight | Widen to 1e-5 or 1e-4 for accumulated operations |
| proptest shrinking takes too long | Input space too large | Add `#[proptest_config(ProptestConfig::with_cases(100))]` |

</troubleshooting>

<references>
— crates/jepa-core/src/types.rs: Existing test examples (6 tests)
— specs/gherkin/features.feature: BDD scenarios with test vectors
— SPECIFICATION.md: Test vectors embedded in RFC sections
</references>
