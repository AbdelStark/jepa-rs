---
name: workspace-development
description: Activate when a task changes workspace structure, crate boundaries, shared types, or code that may ripple across multiple crates. Use this for most new features, refactors, and bug fixes so work lands in the right crate and public API churn stays contained.
prerequisites: cargo, rustfmt, clippy
---

# Workspace Development

<purpose>
Route work to the smallest owning crate, keep shared-contract changes gated, and finish with the correct verification scope for this Cargo workspace.
</purpose>

<context>
- Root `Cargo.toml` defines 6 workspace members. `fuzz/` is a separate nested workspace.
- `jepa-core` is the shared contract layer. All other library crates depend on it; `crates/jepa` depends on every library crate.
- Public re-exports live in each crate `src/lib.rs`. Missing a re-export is a common downstream failure.
- `Cargo.toml` files, `Cargo.lock`, `.github/workflows/`, `scripts/`, and `specs/differential/` are gated paths.
</context>

<procedure>
1. Map the task to the owning crate: shared contracts -> `jepa-core`; vision models -> `jepa-vision`; training orchestration -> `jepa-train`; planning/memory -> `jepa-world`; checkpoints or ONNX -> `jepa-compat`; CLI/TUI -> `jepa`.
2. Keep the fix inside one crate unless tests or public interfaces force a cross-crate change.
3. If the change needs `Cargo.toml`, `Cargo.lock`, or a gated `jepa-core` public contract file, stop and get approval before editing.
4. Add or update tests in the owning crate first. For cross-crate behavior, update producer tests before consumer code.
5. Run the narrowest command that proves the change: `cargo test -p jepa-core`, `cargo test -p jepa-vision`, `cargo test -p jepa-compat`, or `cargo test -p jepa`.
6. Before handoff, run `cargo check --workspace --all-targets`, `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo fmt -- --check`.
</procedure>

<patterns>
<do>
- Reuse existing crate-local patterns before inventing new helper modules.
- Keep cross-crate imports explicit: `use crate::...` within a crate and `use jepa_core::...` across crates.
- Review the owning crate `src/lib.rs` whenever you add a new public type or module.
</do>
<dont>
- Do not move logic into `jepa-core` just because more than one crate might use it; escalate only when a stable shared contract is truly needed.
- Do not change workspace manifests, dependency sets, or feature lists without approval.
- Do not expand a change into `crates/jepa` unless the CLI or TUI surface actually needs to change.
</dont>
</patterns>

<examples>
Example: add a helper inside `jepa-world` without changing shared contracts.
```rust
impl RandomShootingConfig {
    pub fn capped(self, max_horizon: usize) -> Self {
        Self {
            horizon: self.horizon.min(max_horizon),
            ..self
        }
    }
}
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| `unresolved import` after adding a type | The item is not re-exported from `src/lib.rs` | Update the owning crate `src/lib.rs` or import from the concrete module instead |
| Downstream crate breaks after a small core edit | A shared contract changed | Re-check `jepa-core` public types, then run workspace tests |
| Cargo waits on file locks | Multiple cargo jobs are using the same checkout | Serialize cargo commands or wait for the other process to finish |
</troubleshooting>

<references>
- `Cargo.toml`: workspace members and shared dependency versions
- `crates/jepa-core/src/lib.rs`: shared public surface
- `crates/jepa/src/lib.rs`: CLI/TUI entry points
</references>
