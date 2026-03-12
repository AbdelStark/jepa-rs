---
name: implementing-rfcs
description: Guide for implementing SPECIFICATION.md RFCs into working Rust code. Activate when implementing any trait, struct, or module defined in the RFC archive — including encoder, predictor, energy, masking, collapse, EMA, config, training loop, world model, or hierarchical JEPA.
prerequisites: SPECIFICATION.md must be read for the target RFC section
---

# Implementing RFCs

<purpose>
Turn RFC specifications from SPECIFICATION.md into idiomatic, tested Rust code.
Core traits are complete in jepa-core. Remaining work is ViT/predictor implementations and outer crates.
</purpose>

<context>
RFC-to-module mapping:

| RFC     | Module(s)                    | Crate        | Status      |
|---------|------------------------------|--------------|-------------|
| RFC-001 | types.rs, config.rs          | jepa-core    | DONE (12 + 18 tests) |
| RFC-002 | encoder.rs                   | jepa-core + jepa-vision | Trait done (1 test), ViT impl needed |
| RFC-003 | predictor.rs                 | jepa-core    | Trait done (2 tests), cross-attention impl needed |
| RFC-004 | energy.rs                    | jepa-core    | DONE (18 tests, L2/Cosine/SmoothL1) |
| RFC-005 | masking.rs                   | jepa-core    | DONE (14 tests, Block/Spatiotemporal/MultiBlock) |
| RFC-006 | collapse.rs                  | jepa-core    | DONE (21 tests, VICReg/BarlowTwins) |
| RFC-007 | ema.rs                       | jepa-core    | DONE (27 tests, Ema/CosineMomentumSchedule) |
| RFC-008 | trainer.rs, schedule.rs, etc | jepa-train   | Stub crate |
| RFC-009 | action.rs, planner.rs, etc   | jepa-world   | Stub crate |
| RFC-010 | hierarchy.rs, memory.rs      | jepa-world   | Stub crate |

The build compiles cleanly. All 88 unit tests + 6 doc tests pass.
Next targets: ViT encoder (jepa-vision), cross-attention predictor, training loop.
</context>

<procedure>
1. Read the target RFC section in SPECIFICATION.md. Note the exact type signatures, method signatures, and invariants.
2. Read specs/gherkin/features.feature for BDD scenarios related to this RFC.
3. Check existing implementations in jepa-core/src/ for patterns to follow:
   - energy.rs: Trait + multiple implementations pattern
   - masking.rs: Strategy trait + block/spatiotemporal/multi-block variants
   - collapse.rs: Regularizer trait + VICReg/BarlowTwins pattern
   - config.rs: Builder pattern with validation
4. Write tests FIRST (TDD):
   - Use `type TestBackend = burn_ndarray::NdArray<f32>;` in test modules
   - Cover every invariant stated in the RFC
   - Include edge cases: empty inputs, zero-dim, single-element
   - Use proptest for numerical properties (e.g., energy >= 0)
5. Implement until all tests pass
6. Run `cargo test -p [crate]` — all existing tests must still pass
7. Run `cargo clippy --all-targets` — zero warnings
8. Run `cargo fmt -- --check` — must pass

Decision point at step 3: If implementing in an outer crate (jepa-vision, jepa-train), reference jepa-core traits via `jepa_core::` imports. The crate dependencies are already configured.
</procedure>

<patterns>
<do>
  — Follow RFC type signatures exactly — don't redesign the API
  — Use `burn::tensor::Tensor<B, N>` for tensor operations
  — Wrap returned tensors in semantic types (Representation, Energy)
  — Include test vectors from the RFC as literal test cases
  — Add `#[derive(Debug, Clone)]` to all public structs
  — Use `thiserror::Error` for error enums
  — Study the existing implementations (energy.rs, masking.rs, collapse.rs) for patterns
</do>
<dont>
  — Don't implement features beyond what the RFC specifies
  — Don't use `unwrap()` in library code — return Result
  — Don't hardcode tensor dimensions — derive from config/input
  — Don't skip the TDD loop — tests come before implementation
  — Don't modify existing passing tests without justification
</dont>
</patterns>

<examples>
Pattern: Implementing a new variant of an existing trait (from energy.rs)

```rust
use burn::tensor::{backend::Backend, Tensor};
use crate::types::{Representation, Energy};

/// Smooth L1 (Huber) energy function.
/// See SPECIFICATION.md RFC-004.
pub struct SmoothL1Energy {
    pub beta: f32,
}

impl<B: Backend> EnergyFn<B> for SmoothL1Energy {
    fn compute(&self, predicted: &Representation<B>, target: &Representation<B>) -> Energy<B> {
        // Implementation using burn tensor ops
    }
}
```

Pattern: Builder with validation (from config.rs)

```rust
pub struct JepaConfigBuilder { /* fields with Option<T> */ }
impl JepaConfigBuilder {
    pub fn build(self) -> Result<JepaConfig, ConfigError> {
        let config = JepaConfig { /* ... */ };
        config.validate()?;
        Ok(config)
    }
}
```
</examples>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| E0432 unresolved import after adding type | Type name doesn't match re-export in lib.rs | Check lib.rs `pub use` statement matches exactly |
| burn tensor operation not found | Wrong tensor rank or missing Backend bound | Check burn 0.16 API — operations are rank-specific |
| Test can't construct Representation | Need burn-ndarray in dev-dependencies | Already in jepa-core; check crate-level Cargo.toml for outer crates |
| Existing tests fail after changes | Public API regression | Check that trait signatures haven't changed |

</troubleshooting>

<references>
— SPECIFICATION.md: Complete RFC archive (1105 lines)
— specs/gherkin/features.feature: 27 BDD scenarios with test vectors
— crates/jepa-core/src/energy.rs: Reference for trait + multiple implementations
— crates/jepa-core/src/collapse.rs: Reference for complex numerical implementations
— crates/jepa-core/src/config.rs: Reference for builder pattern with validation
— crates/jepa-core/src/lib.rs: Re-export declarations
</references>
