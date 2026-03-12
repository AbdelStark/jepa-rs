# Multi-Agent Orchestration for jepa-rs

<project>
jepa-rs is an alpha Rust workspace for JEPA components on top of burn 0.16.

Current state as of March 12, 2026:
- Workspace build, tests, clippy, and docs pass locally.
- Safetensors support is functional.
- ONNX loading is still a stub.
- The generic trainer does not yet enforce strict pre-encoder masking semantics.
</project>

<roles>

| Role | Responsibility | Boundaries |
|------|----------------|------------|
| Orchestrator | Plan work, split tasks, manage conflicts | Does not write implementation code |
| Implementer | Modify library code and tests | Does not change public trait signatures without approval |
| Reviewer | Audit correctness, numerical behavior, safety | Does not implement fixes directly |
| Tester | Extend tests, fixtures, and differential checks | Restrict changes to tests and fixtures where possible |

</roles>

<shared_context>

Read first:
- [`README.md`](./README.md)
- [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- [`PRODUCTION_GAP.md`](./PRODUCTION_GAP.md)
- [`ROADMAP.md`](./ROADMAP.md)
- [`WORK_PACKAGES.md`](./WORK_PACKAGES.md)
- [`SPECIFICATION.md`](./SPECIFICATION.md)

Critical crates:
- `crates/jepa-core`: shared contracts
- `crates/jepa-vision`: vision encoders and predictor
- `crates/jepa-train`: training orchestration
- `crates/jepa-compat`: safetensors and ONNX adapter
- planning source of truth: `PRODUCTION_GAP.md`, `ROADMAP.md`, `WORK_PACKAGES.md`

</shared_context>

<parallel_execution>

Safe to parallelize:
- documentation updates
- test additions
- `jepa-compat` work that does not require dependency changes
- benchmarks, fuzz targets, and differential fixtures

Serialize these:
- any change to `jepa-core` public traits
- any `lib.rs` re-export changes
- any `Cargo.toml` change

Conflict protocol:
1. Declare target files before editing.
2. If another agent already owns a file, wait.
3. Treat `Cargo.toml` edits as human-gated work.

</parallel_execution>

<task_template>

```text
## Task: [short description]

Objective:
- [specific deliverable]

Context:
- Read the relevant RFC in SPECIFICATION.md
- Read ARCHITECTURE.md if the change crosses crate boundaries
- Reuse patterns from jepa-core where possible

Acceptance criteria:
- [ ] Code matches the current implementation constraints
- [ ] Tests cover the new or fixed behavior
- [ ] `cargo test` passes
- [ ] `cargo clippy --all-targets -- -D warnings` passes
- [ ] `cargo fmt -- --check` passes

Constraints:
- Do not change public trait signatures without approval
- Do not modify SPECIFICATION.md
- Do not add dependencies without approval
```

</task_template>

<gotchas>

- `JepaComponents::forward_step` is not a strict masked-encoder trainer. Do not design downstream work as if target tokens are hidden before encoder attention.
- `TransformerPredictor` now expects real flattened token indices in `target_positions`.
- `Representation::gather` preserves masks. Rely on that instead of rebuilding token masks by hand.
- ONNX tasks that need real parsing or runtime execution require a dependency addition and must be escalated.

</gotchas>

<escalation>

Escalate when:
- a public trait signature must change
- a dependency must be added or updated
- strict masked-encoder semantics require an architectural change across crates
- a performance fix conflicts with correctness or clarity

Format:

```text
**ESCALATION**: [one-line summary]
**Context**: [crate/module]
**Blocker**: [specific issue]
**Options**:
1. [Option] — Tradeoff: [gain/lose]
2. [Option] — Tradeoff: [gain/lose]
**Recommendation**: [best option]
```

</escalation>
