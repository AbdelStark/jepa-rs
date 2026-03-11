# Multi-Agent Orchestration for jepa-rs

<applicability>
This project benefits from multi-agent coordination due to:
- 5-crate workspace with independent implementation paths
- Each RFC can be implemented in parallel once dependencies are resolved
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

## Task Decomposition Strategy

The JEPA implementation has a natural dependency graph:

```
RFC-001 (types, config) ← DONE (types), PENDING (config)
    ↓
RFC-004 (energy) ─────────────────┐
RFC-005 (masking) ────────────────┤
RFC-006 (collapse/VICReg) ────────┤
RFC-007 (EMA) ────────────────────┤
    ↓                              ↓
RFC-002 (encoder) ──────→ RFC-003 (predictor)
                              ↓
                    RFC-008 (training loop)
                              ↓
              RFC-009 (world model) ← RFC-010 (hierarchical)
```

**Parallelizable immediately** (no inter-RFC dependencies):
- RFC-004: Energy functions (jepa-core/energy.rs)
- RFC-005: Masking strategies (jepa-core/masking.rs)
- RFC-006: Collapse prevention (jepa-core/collapse.rs)
- RFC-007: EMA (jepa-core/ema.rs)
- RFC-001 remainder: JepaConfig (jepa-core/config.rs)

**Must serialize**:
- RFC-002 (encoder) depends on types + config
- RFC-003 (predictor) depends on encoder + types
- RFC-008 (training) depends on all core traits
- RFC-009/010 depend on training loop

</delegation_protocol>

<task_format>

Every delegated implementation task must include:

```
## Task: Implement [RFC-XXX component]

**Objective**: [What "done" looks like — specific types/traits defined and tested]

**Context**:
- Read first: SPECIFICATION.md section for RFC-XXX
- Read: specs/gherkin/features.feature (relevant scenarios)
- Modify: crates/jepa-core/src/[module].rs
- Verify: crates/jepa-core/src/lib.rs re-export resolves

**Acceptance criteria**:
- [ ] Type/trait defined matching RFC specification
- [ ] Unit tests covering all RFC test vectors
- [ ] Property tests for numerical invariants
- [ ] `cargo test -p jepa-core` passes
- [ ] `cargo clippy -p jepa-core` — zero warnings
- [ ] Re-export in lib.rs resolves without E0432

**Constraints**:
- Do NOT modify types.rs (already implemented)
- Do NOT change the trait name (must match lib.rs re-export)
- Use `B: Backend` generic for all tensor-bearing types
```

</task_format>

<parallel_execution>

Safe to parallelize:
- RFC-004 + RFC-005 + RFC-006 + RFC-007 (independent modules, no shared mutable state)
- Unit tests + BDD scenario implementation (different file sets)
- Documentation + implementation (different files)

Must serialize:
- Any change to types.rs (shared dependency for all modules)
- Changes to lib.rs re-exports (single file, conflict-prone)
- Cargo.toml dependency changes (workspace-wide impact)
- Config implementation if other modules depend on it

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
