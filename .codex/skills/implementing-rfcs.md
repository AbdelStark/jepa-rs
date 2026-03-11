---
name: implementing-rfcs
description: Guide for implementing SPECIFICATION.md RFCs into working Rust code. Activate when implementing any trait, struct, or module defined in the RFC archive — including encoder, predictor, energy, masking, collapse, EMA, config, training loop, world model, or hierarchical JEPA.
prerequisites: SPECIFICATION.md must be read for the target RFC section
---

# Implementing RFCs

<purpose>
Turn RFC specifications from SPECIFICATION.md into idiomatic, tested Rust code.
Each placeholder module in jepa-core maps to one or more RFCs.
</purpose>

<context>
RFC-to-module mapping:

| RFC     | Module(s)                    | Crate        | Status      |
|---------|------------------------------|--------------|-------------|
| RFC-001 | types.rs, config.rs          | jepa-core    | types: done, config: placeholder |
| RFC-002 | encoder.rs                   | jepa-core + jepa-vision | placeholder |
| RFC-003 | predictor.rs                 | jepa-core    | placeholder |
| RFC-004 | energy.rs                    | jepa-core    | placeholder |
| RFC-005 | masking.rs                   | jepa-core    | placeholder |
| RFC-006 | collapse.rs                  | jepa-core    | placeholder |
| RFC-007 | ema.rs                       | jepa-core    | placeholder |
| RFC-008 | (training loop)              | jepa-train   | stub        |
| RFC-009 | (action-conditioned world)   | jepa-world   | stub        |
| RFC-010 | (hierarchical JEPA)          | jepa-core    | stub        |

All placeholder modules currently contain only a doc comment.
lib.rs re-exports types that don't exist yet — this causes compile errors.
</context>

<procedure>
1. Read the target RFC section in SPECIFICATION.md. Note the exact type signatures, method signatures, and invariants.
2. Read specs/gherkin/features.feature for BDD scenarios related to this RFC.
3. Define the trait/struct in the placeholder module file:
   - Use `B: Backend` generic for all tensor-bearing types
   - Add doc comments with `/// ` citing the RFC number
   - Match the API surface from the RFC exactly
4. Write tests FIRST (TDD):
   - Use `type TestBackend = burn_ndarray::NdArray<f32>;` in test modules
   - Cover every invariant stated in the RFC
   - Include edge cases: empty inputs, zero-dim, single-element
   - Use proptest for numerical properties (e.g., energy >= 0)
5. Implement until all tests pass
6. Verify the re-export in lib.rs now resolves (the E0432 error for this type should disappear)
7. Run full workspace: `cargo test && cargo clippy --all-targets`

Decision point at step 3: If the RFC references types from another unimplemented RFC, implement the dependency first or use a minimal stub.
</procedure>

<patterns>
<do>
  — Follow RFC type signatures exactly — don't redesign the API
  — Use `burn::tensor::Tensor<B, N>` for tensor operations
  — Wrap returned tensors in semantic types (Representation, Energy)
  — Include test vectors from the RFC as literal test cases
  — Add `#[derive(Debug, Clone)]` to all public structs
  — Use `thiserror::Error` for error enums
</do>
<dont>
  — Don't implement features beyond what the RFC specifies
  — Don't use `unwrap()` in library code — return Result
  — Don't hardcode tensor dimensions — derive from config/input
  — Don't skip the TDD loop — tests come before implementation
</dont>
</patterns>

<examples>
Example: Implementing a trait from RFC (pattern from types.rs)

```rust
use burn::tensor::{backend::Backend, Tensor};
use crate::types::{Representation, Energy};

/// Energy function that measures compatibility between representations.
/// See SPECIFICATION.md RFC-004.
pub trait EnergyFn<B: Backend> {
    /// Compute energy between predicted and target representations.
    /// Lower energy = better prediction.
    fn compute(&self, predicted: &Representation<B>, target: &Representation<B>) -> Energy<B>;
}
```
</examples>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| E0432 unresolved import after adding type | Type name doesn't match re-export in lib.rs | Check lib.rs `pub use` statement matches exactly |
| burn tensor operation not found | Wrong tensor rank or missing Backend bound | Check burn 0.16 API — operations are rank-specific |
| Test can't construct Representation | Need burn-ndarray in dev-dependencies | Add to [dev-dependencies] in crate's Cargo.toml |

</troubleshooting>

<references>
— SPECIFICATION.md: Complete RFC archive (1105 lines)
— specs/gherkin/features.feature: BDD scenarios with test vectors
— crates/jepa-core/src/types.rs: Reference implementation pattern
— crates/jepa-core/src/lib.rs: Re-export declarations (must match)
</references>
