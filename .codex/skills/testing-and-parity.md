---
name: testing-and-parity
description: Activate when the task mentions tests, regressions, CI, parity, coverage, property tests, or fuzzing. Use this skill for any bug fix or behavior change that needs proof, especially around masking, strict image parity, or checkpoint parsing.
prerequisites: cargo, cargo-fuzz (nightly for fuzz smoke)
---

# Testing and Parity

<purpose>
Choose the smallest test surface that proves the change, then expand to the repo's required quality gates without skipping strict parity when it matters.
</purpose>

<context>
- Unit tests live beside code in each crate. Cross-cutting vision checks live in `crates/jepa-vision/tests/integration.rs`.
- `cargo test --workspace` runs unit, integration, and doc tests for this repo.
- Strict image parity is fixture-driven. The runner is `scripts/run_parity_suite.sh`.
- Fuzzing lives in the nested `fuzz/` workspace with targets `masking`, `gather`, `energy`, and `checkpoint_parsing`.
- CI gates in `.github/workflows/ci.yml` are: check, test, clippy, fmt, doc, coverage, bench smoke, parity, package smoke, fuzz smoke, and security audit.
</context>

<procedure>
1. Start with the narrowest useful command: `cargo test -p jepa-core`, `cargo test -p jepa-vision`, `cargo test -p jepa-compat`, or `cargo test -p jepa`.
2. Add regression tests in the same file as the changed code unless the behavior is cross-crate or parity-specific.
3. For strict image semantics, run `scripts/run_parity_suite.sh` after the targeted crate tests are green.
4. If a change affects public behavior or shared types, run the full workspace gates: `cargo check --workspace --all-targets`, `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt -- --check`.
5. Consider `fuzz/` only for parser, checkpoint, masking, or gather logic. Treat it as a separate nightly-only validation layer.
</procedure>

<patterns>
<do>
- Use `burn_ndarray::NdArray<f32>` on CPU for deterministic tests unless the behavior requires another backend.
- Keep float tolerances tight enough to catch regressions. Use wider tolerances only when you can justify accumulated numeric error.
- Keep ignored parity tests ignored and run them through the script runner.
</do>
<dont>
- Do not delete `proptest-regressions/` artifacts to make a failure disappear.
- Do not skip workspace-wide verification after changing shared contracts or CLI behavior.
- Do not rewrite parity fixtures or their tolerances without approval.
</dont>
</patterns>

<examples>
Example: standard verification sequence for a strict image-path fix.
```bash
cargo test -p jepa-vision
scripts/run_parity_suite.sh
cargo test --workspace
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| `test_ijepa_strict_fixture_parity` shows as ignored | The parity test is script-driven | Run `scripts/run_parity_suite.sh` |
| `no parity fixtures found` | The runner was pointed at the wrong directory or repo root is wrong | Run the script from repo root or pass a valid fixture path |
| Cargo blocks on file locks | Multiple cargo jobs are running in one checkout | Serialize the jobs and rerun |
</troubleshooting>

<references>
- `.github/workflows/ci.yml`: exact CI gates
- `crates/jepa-vision/tests/integration.rs`: strict parity integration point
- `specs/differential/README.md`: fixture workflow and tolerance policy
- `fuzz/Cargo.toml`: fuzz targets and nightly tooling
</references>
