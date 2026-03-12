# Multi-Agent Orchestration for jepa-rs

<applicability>
This project benefits from multi-agent coordination due to:
- 5-crate workspace with clear separation of concerns
- Testing layers (unit, BDD, differential, fuzz) are independently developable
- Enhancement work spans multiple independent crates
- Clear separation: core traits vs vision vs world vs training vs compat
</applicability>

<roles>

| Role         | Model Tier | Responsibility                                   | Boundaries                             |
|--------------|------------|--------------------------------------------------|----------------------------------------|
| Orchestrator | Frontier   | Plan enhancement tasks, decompose work            | NEVER writes implementation code       |
| Implementer  | Mid-tier   | Implement features/tests per RFC, extend tests    | NEVER changes public API without ask   |
| Reviewer     | Frontier   | Validate correctness, numerical accuracy, safety  | NEVER implements fixes (sends back)    |
| Tester       | Mid-tier   | Write tests, run differential checks, fuzz        | Only operates on test code/fixtures    |

</roles>

<delegation_protocol>

## Implementation Status & Dependency Graph

All 10 RFCs from SPECIFICATION.md are **fully implemented and tested**.

```
RFC-001 (types, config)          ← DONE (types.rs, config.rs — 30 tests)
RFC-002 (encoder + ViT)          ← DONE (encoder.rs, vit.rs, patch.rs, rope.rs — 20+ tests)
RFC-003 (predictor + I-JEPA)     ← DONE (predictor.rs, image.rs, video.rs — 27+ tests)
RFC-004 (energy functions)       ← DONE (energy.rs: L2, Cosine, SmoothL1 — 18 tests)
RFC-005 (masking strategies)     ← DONE (masking.rs: Block, Spatiotemporal, MultiBlock — 14 tests)
RFC-006 (collapse prevention)    ← DONE (collapse.rs: VICReg, BarlowTwins — 21 tests)
RFC-007 (EMA target encoder)     ← DONE (ema.rs: Ema, CosineMomentumSchedule — 27 tests)
RFC-008 (training loop)          ← DONE (trainer.rs, schedule.rs, checkpoint.rs, step.rs — 21 tests)
RFC-009 (world model)            ← DONE (action.rs, planner.rs — 10 tests)
RFC-010 (hierarchical JEPA)      ← DONE (hierarchy.rs, memory.rs — 13 tests)
```

**Remaining enhancement work** (all parallelizable):
- ONNX runtime integration (jepa-compat/onnx.rs — needs `ort` dependency)
- BDD test wiring (specs/gherkin → step definitions)
- Differential testing against Python reference implementations
- Fuzz testing targets for masking and energy functions
- Additional benchmarks and profiling

</delegation_protocol>

<task_format>

Every delegated enhancement task must include:

```
## Task: [Enhancement description]

**Objective**: [What "done" looks like — specific deliverables]

**Context**:
- Read first: SPECIFICATION.md section for relevant RFC
- Read: specs/gherkin/features.feature (relevant scenarios)
- Modify: crates/[crate]/src/[module].rs
- Reference: crates/jepa-core/src/ for established patterns

**Acceptance criteria**:
- [ ] Enhancement implemented matching specification
- [ ] Tests added covering new functionality
- [ ] Property tests for numerical invariants (where applicable)
- [ ] `cargo test` passes (all 261+ existing tests must still pass)
- [ ] `cargo clippy --all-targets` — zero warnings
- [ ] Doc tests pass if doc examples are added

**Constraints**:
- Do NOT modify existing public trait signatures without approval
- Do NOT change SPECIFICATION.md (read-only)
- Use `B: Backend` generic for all tensor-bearing types
```

</task_format>

<parallel_execution>

Safe to parallelize:
- ONNX runtime integration (jepa-compat) — independent crate
- BDD test wiring (specs/) — test-only changes
- Differential test fixtures — test-only changes
- Fuzz targets — test-only changes
- Documentation improvements — non-code changes
- Benchmark additions — independent of library code

Must serialize:
- Any change to jepa-core public traits (shared dependency for all crates)
- Changes to lib.rs re-exports (single file, conflict-prone)
- Cargo.toml dependency changes (workspace-wide impact)

Conflict protocol:
1. Before starting, declare which files will be modified
2. If two agents need the same file, lower-priority agent waits
3. lib.rs changes are atomic — only one agent modifies re-exports at a time

</parallel_execution>

<escalation>

Escalate to human when:
- A public trait signature needs to change
- burn 0.16 API doesn't support a required operation
- Performance requirement conflicts with correctness requirement
- New dependency additions to workspace Cargo.toml

Escalation format:
```
**ESCALATION**: [one-line summary]
**Context**: [which module/RFC]
**Blocker**: [specific issue]
**Options**:
1. [Option] — Tradeoff: [gain/lose]
2. [Option] — Tradeoff: [gain/lose]
**Recommendation**: [which and why]
```

</escalation>
