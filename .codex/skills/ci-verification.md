---
name: ci-verification
description: CI pipeline, quality gates, and release readiness for jepa-rs. Activate when discussing CI failures, release processes, coverage requirements, fuzz testing, or benchmark budgets.
prerequisites: .github/workflows/ci.yml exists
---

# CI & Verification

<purpose>
Understand and work with the 11-job CI pipeline that enforces correctness, quality, and release readiness.
CI triggers on push to main and PRs to main.
</purpose>

<context>
CI jobs (all on ubuntu-latest, RUSTFLAGS=-D warnings):

| Job | Command | Gate |
|-----|---------|------|
| check | `cargo check --workspace --all-targets` | Must pass |
| test | `cargo test --workspace` | 356 tests, must pass |
| clippy | `cargo clippy --workspace --all-targets -- -D warnings` | Zero warnings |
| fmt | `cargo fmt -- --check` | Must pass |
| doc | `cargo doc --no-deps --all-features` (RUSTDOCFLAGS: -D warnings) | Must pass |
| coverage | `cargo llvm-cov --fail-under-lines 80` | 80% line coverage floor |
| bench-smoke | `cargo bench --workspace --no-run` | Benchmarks must compile |
| parity | `scripts/run_parity_suite.sh` | 3 I-JEPA image fixtures |
| package-smoke | `cargo package` per crate in dependency order | Must package cleanly |
| fuzz | `cargo fuzz run [target] -- -runs=256` (nightly) | masking, gather, energy, checkpoint_parsing |
| audit | `rustsec/audit-check@v2.0.0` | No known vulnerabilities |
</context>

<procedure>
Before pushing code:

1. `cargo fmt -- --check` — fix formatting first (fastest gate)
2. `cargo clippy --all-targets -- -D warnings` — fix all warnings
3. `cargo test --workspace` — all 356 tests must pass
4. `cargo doc --no-deps` — documentation must build cleanly

If CI fails:

1. Check which job failed — the job name tells you the domain
2. For test failures: reproduce locally with `cargo test -p [crate] -- --nocapture`
3. For clippy: run `cargo clippy --all-targets -- -D warnings` locally
4. For coverage: run `cargo llvm-cov` if available, or add tests for uncovered paths
5. For parity: check if fixtures need updating after intentional behavior changes
6. For fuzz: reproduce with `cargo fuzz run [target]`, check the crash input

Release readiness (from PRODUCTION_GAP.md):

1. All P0/P1 gaps closed (currently: all closed except crates.io publish)
2. Package smoke passes: `cargo package` for each crate in dependency order
3. Local release-candidate rehearsal complete (already done)
4. crates.io publish requires maintainer approval
</procedure>

<patterns>
<do>
  — Run fmt and clippy before pushing — they're the fastest feedback
  — Write tests that maintain the 80% coverage floor
  — Check parity fixtures if changing strict forward paths
  — Add fuzz targets for new unbounded/parsing code
</do>
<dont>
  — Don't disable CI checks to make a build pass
  — Don't lower the coverage floor without approval
  — Don't modify CI pipeline without approval (gated file)
  — Don't skip the package-smoke step before release work
</dont>
</patterns>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| Coverage below 80% | New untested code paths | Add unit tests for uncovered branches |
| Parity test fails | Strict forward path behavior changed | Check if change is intentional; update fixtures if so |
| Fuzz crash | Edge case in masking/gather/energy/checkpoint | Reproduce locally, add regression test, fix the bug |
| Package-smoke fails | Missing metadata or invalid Cargo.toml | Check package metadata (license, description, readme) |
| Audit fails | Known vulnerability in dependency | Check advisory, update dependency if fix available |

</troubleshooting>

<references>
— .github/workflows/ci.yml: Full CI pipeline definition (11 jobs)
— scripts/run_parity_suite.sh: Differential parity test runner
— fuzz/: Fuzz target definitions
— PRODUCTION_GAP.md: Gap register and release criteria
— ROADMAP.md: Milestone tracking
</references>
