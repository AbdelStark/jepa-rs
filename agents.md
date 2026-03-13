<overview>
Use this file only when a task benefits from parallel work. If a single agent is active, treat it as a decomposition and review policy.
</overview>

<roles>

| Role | Model Tier | Responsibility | Boundaries |
|------|------------|----------------|------------|
| Orchestrator | frontier | Decompose work, assign file ownership, review integration risk | Never lands implementation code directly |
| Implementer | frontier | Make crate-scoped code changes and run local verification | Never edits gated paths without approval |
| Specialist | frontier | Handle domain-heavy work in `jepa-vision`, `jepa-compat`, or `crates/jepa` | Only works inside its declared files |
| Reviewer | frontier | Check correctness, regressions, parity impact, and boundary violations | Never fixes issues in the same pass; sends work back |

</roles>

<shared_context>
Read `CLAUDE.md` first. Then load only the relevant repo-local skill from `.codex/skills/`.

Common routing:
- Workspace or cross-crate change: `workspace-development.md`
- Vision or masking change: `strict-vision-models.md`
- Test, CI, or parity work: `testing-and-parity.md`
- safetensors or ONNX work: `checkpoint-and-onnx.md`
- CLI, demos, or TUI work: `cli-and-demos.md`
</shared_context>

<delegation_protocol>
1. Analyze the task and assign an owning crate before any edits start.
2. Compare target file sets for overlap.
3. Delegate routine crate-local work to an Implementer.
4. Delegate parity-sensitive, ONNX, or TUI work to a Specialist when those domains dominate the task.
5. Reserve architectural or boundary decisions for the Orchestrator.
6. Require a Reviewer pass whenever the task touches `jepa-core`, strict parity fixtures, manifests, or public CLI behavior.
7. Integrate only after all assigned file owners report green targeted tests.
</delegation_protocol>

<task_format>
```text
## Task: [clear title]

Objective:
- [what done looks like]

Context:
- Files to read: [exact paths]
- Files to modify: [exact paths]
- Related crates or types: [exact names]

Acceptance criteria:
- [ ] Targeted tests pass: [exact command]
- [ ] Workspace checks pass if shared behavior changed
- [ ] No gated path was edited without approval

Constraints:
- Do not modify: [out-of-scope paths]
- Escalate if: [public API, manifests, parity fixtures, CI, scripts]

Handoff:
- Report changed files, commands run, and residual risk
```
</task_format>

<state_machine>
`PENDING -> ASSIGNED -> IN_PROGRESS -> REVIEW -> APPROVED -> DONE`

Alternative exits:
- `IN_PROGRESS -> BLOCKED` when the agent reports what failed, what was tried, and what input is needed
- `REVIEW -> REJECTED` when the reviewer names exact defects and the expected fix
- `BLOCKED -> ESCALATED` when the blocker is a gated edit or unresolved design conflict
</state_machine>

<parallel_execution>
Safe to parallelize:
- Different crates with no shared file ownership
- Docs and skill updates
- New tests in different crates
- `jepa-world` work alongside `jepa-compat` or `crates/jepa`

Must serialize:
- `Cargo.toml`, `Cargo.lock`, `.github/workflows/ci.yml`
- `scripts/` and `specs/differential/`
- `crates/jepa-core/src/lib.rs` and other gated public contract files
- Any task that changes clap flags in `crates/jepa/src/cli.rs` and the corresponding command implementation

Conflict protocol:
1. Detect overlap before editing.
2. Give priority to the task with the smaller, more central file set.
3. Wait for that task to land.
4. Rebase, re-run targeted tests, then continue.
</parallel_execution>

<review_gates>
Reviewer checklist:
- Public behavior matches strict versus approximate semantics honestly
- New or changed CLI flags have clap tests
- Mask flow and target positions are covered by tests when touched
- `cargo check --workspace --all-targets`, `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo fmt -- --check` ran when needed
- `scripts/run_parity_suite.sh` ran for strict image-path changes
</review_gates>

<escalation>
Escalate to a human when:
- A `Cargo.toml` or `Cargo.lock` edit is required
- A `jepa-core` public contract or re-export surface must change
- A parity fixture or tolerance must change
- CI workflow or script changes are needed
- Confidence drops below 70 percent on a behavior-changing decision

Escalation format:
```text
**ESCALATION**: [one-line summary]
**Context**: [crate or path]
**Blocker**: [specific issue]
**Options**:
1. [option] - Tradeoff: [gain and cost]
2. [option] - Tradeoff: [gain and cost]
**Recommendation**: [best option]
```
</escalation>
