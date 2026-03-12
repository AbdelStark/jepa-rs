# Multi-Agent Orchestration for jepa-rs

<applicability>
This project benefits from multi-agent coordination due to:
- 5-crate workspace with independent implementation paths
- Remaining RFCs (002, 003, 008, 009, 010) span multiple crates
- Testing layers (unit, BDD, differential, fuzz) are independently developable
- Clear separation: core traits vs vision vs world vs training vs compat
</applicability>

<roles>

| Role         | Model Tier | Responsibility                                   | Boundaries                             |
|--------------|------------|--------------------------------------------------|----------------------------------------|
| Orchestrator | Frontier   | Plan RFC implementation order, decompose tasks   | NEVER writes implementation code       |
| Implementer  | Mid-tier   | Implement traits/structs per RFC, write tests     | NEVER changes public API without ask   |
| Reviewer     | Frontier   | Validate correctness, numerical accuracy, safety  | NEVER implements fixes (sends back)    |
| Tester       | Mid-tier   | Write tests, run differential checks, fuzz        | Only operates on test code/fixtures    |

</roles>

<delegation_protocol>

## Implementation Status & Dependency Graph

```
RFC-001 (types, config)          ← DONE (types.rs, config.rs)
RFC-004 (energy functions)       ← DONE (energy.rs: L2, Cosine, SmoothL1)
RFC-005 (masking strategies)     ← DONE (masking.rs: Block, Spatiotemporal, MultiBlock)
RFC-006 (collapse prevention)    ← DONE (collapse.rs: VICReg, BarlowTwins)
RFC-007 (EMA target encoder)     ← DONE (ema.rs: Ema, CosineMomentumSchedule)
    ↓
RFC-002 (encoder)                ← PARTIAL (trait done; ViT impl needed in jepa-vision)
    ↓
RFC-003 (predictor)              ← PARTIAL (trait done; cross-attention impl needed)
    ↓
RFC-008 (training loop)          ← STUB (jepa-train crate)
    ↓
RFC-009 (world model)            ← STUB (jepa-world crate)
RFC-010 (hierarchical JEPA)      ← STUB (jepa-world crate)
```

**Parallelizable now** (independent crates/modules):
- RFC-002 ViT implementation (jepa-vision: vit.rs, patch.rs, rope.rs)
- RFC-003 cross-attention predictor (jepa-core or jepa-vision)
- jepa-compat checkpoint loading (safetensors.rs, keymap.rs)
- BDD test wiring (specs/gherkin → step definitions)
- Differential test fixtures (Python reference generation)

**Must serialize**:
- RFC-008 (training loop) depends on encoder + predictor implementations
- RFC-009/010 depend on training loop
- Any changes to jepa-core public traits (shared dependency)

</delegation_protocol>

<task_format>

Every delegated implementation task must include:

```
## Task: Implement [RFC-XXX component]

**Objective**: [What "done" looks like — specific types/traits defined and tested]

**Context**:
- Read first: SPECIFICATION.md section for RFC-XXX
- Read: specs/gherkin/features.feature (relevant scenarios)
- Modify: crates/[crate]/src/[module].rs
- Reference: crates/jepa-core/src/ for established patterns

**Acceptance criteria**:
- [ ] Type/trait/struct defined matching RFC specification
- [ ] Unit tests covering all RFC test vectors
- [ ] Property tests for numerical invariants (where applicable)
- [ ] `cargo test -p [crate]` passes (all 88+ existing tests must still pass)
- [ ] `cargo clippy --all-targets` — zero warnings
- [ ] Doc tests pass if doc examples are added

**Constraints**:
- Do NOT modify existing jepa-core implementations without approval
- Do NOT change public trait signatures (already tested)
- Use `B: Backend` generic for all tensor-bearing types
```

</task_format>

<parallel_execution>

Safe to parallelize:
- ViT encoder (jepa-vision) + cross-attention predictor (independent modules)
- jepa-compat checkpoint loading + BDD test wiring (different file sets)
- Documentation + implementation (different files)
- Unit tests + differential test fixture generation

Must serialize:
- Any change to jepa-core public traits (shared dependency for all crates)
- Changes to lib.rs re-exports (single file, conflict-prone)
- Cargo.toml dependency changes (workspace-wide impact)
- Training loop implementation depends on encoder + predictor

Conflict protocol:
1. Before starting, declare which files will be modified
2. If two agents need the same file, lower-priority agent waits
3. lib.rs changes are atomic — only one agent modifies re-exports at a time

</parallel_execution>

<escalation>

Escalate to human when:
- RFC specification is ambiguous or contradictory
- Numerical tests fail and the correct behavior is unclear
- A public trait signature needs to differ from the RFC
- burn 0.16 API doesn't support an operation the RFC requires
- Performance requirement conflicts with correctness requirement

Escalation format:
```
**ESCALATION**: [one-line summary]
**RFC**: [which RFC section]
**Blocker**: [specific issue]
**Options**:
1. [Option] — Tradeoff: [gain/lose]
2. [Option] — Tradeoff: [gain/lose]
**Recommendation**: [which and why]
```

</escalation>
