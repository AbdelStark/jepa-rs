# Release Process

## Release Goal

A release is publishable only when the workspace matches the real verification
surface described in [`docs/QUALITY_GATES.md`](./QUALITY_GATES.md).
Operational troubleshooting lives in [`docs/OPERATIONS.md`](./OPERATIONS.md),
benchmark-budget policy lives in [`docs/PERFORMANCE.md`](./PERFORMANCE.md), and
the current release-candidate draft notes live in
[`docs/releases/0.1.0-rc-rehearsal.md`](./releases/0.1.0-rc-rehearsal.md).

## Rehearsal Status

As of March 13, 2026, the dependency-order `cargo package` rehearsal succeeds
locally for all six crates. This is a dry-run readiness signal, not a publish
event.

## Pre-Release Checklist

- `cargo fmt -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `cargo test --doc`
- `cargo doc --workspace --no-deps`
- `scripts/run_parity_suite.sh`
- `cargo llvm-cov --workspace --all-features --fail-under-lines 80`
- Fuzz smoke runs complete for all targets
- Benchmarks compile, and any significant regression is reviewed
- README, changelog, and crate metadata reflect current capabilities

## Publishability Checks

Run these before calling a release candidate publishable:

```bash
cargo package -p jepa-core --no-verify
cargo package -p jepa-vision --no-verify --exclude-lockfile
cargo package -p jepa-world --no-verify --exclude-lockfile
cargo package -p jepa-train --no-verify --exclude-lockfile
cargo package -p jepa-compat --no-verify --exclude-lockfile
cargo package -p jepa --no-verify --exclude-lockfile
```

Notes:

- Downstream workspace dependencies now carry explicit version requirements for publishable manifests.
- `--exclude-lockfile` is intentional for the downstream crates while the workspace is still unpublished on crates.io. Without it, Cargo attempts to resolve `jepa-core` from the registry when generating a lockfile for the packaged crate.
- After publishing `jepa-core`, maintainers may rerun downstream `cargo package` or `cargo publish --dry-run` commands without `--exclude-lockfile` if they want to confirm registry resolution before publishing the remaining crates.
- For a local rehearsal on an in-progress branch, `--allow-dirty` is acceptable. The actual tagged release path should run from a clean checkout.

## Versioning Policy

- Before `1.0`, breaking API changes are allowed but must be called out in the
  changelog and release notes.
- After `1.0`, semantic versioning applies at the public API surface.
- Trait-signature changes in `jepa-core` are release-note material, even before
  `1.0`, because they cascade through the whole workspace.

## Crate Publish Order

Publish in dependency order:

1. `jepa-core`
2. `jepa-vision`
3. `jepa-world`
4. `jepa-train`
5. `jepa-compat`
6. `jepa`

## Support Policy

- Supported status must be stated explicitly in the README and release notes.
- If ONNX loading or parity coverage has limits, those limits must be written
  down for the released version.
- A release is not called production-grade unless the P0 and P1 gaps in
  `PRODUCTION_GAP.md` are closed.
- For the first release candidate, the ONNX boundary is metadata inspection and initializer loading only. Full ONNX runtime execution is a no-go unless maintainers approve new scope and dependencies first.

## Release Steps

1. Update version numbers and changelog entries.
2. Run the full verification surface locally, including `scripts/run_parity_suite.sh`.
3. Run the publishability checks in dependency order.
4. Prepare release notes from [`docs/releases/0.1.0-rc-rehearsal.md`](./releases/0.1.0-rc-rehearsal.md) and confirm the known-limitations section still matches the code.
5. Tag the release candidate in git.
6. Publish crates in dependency order.
7. Rebuild docs and verify docs.rs metadata.
8. Announce the release with status, scope, and known limitations.

## Rollback And Partial Publish

Crates.io releases are effectively append-only. If a publish attempt fails
mid-sequence:

1. Stop the publish sequence immediately.
2. Record exactly which crate versions were published and which were not.
3. If a published crate is unusable or points at a broken dependency chain, yank that version on crates.io.
4. Fix the issue in git, cut a new candidate version, and rerun the verification and packaging flow from the start.
5. Update the release notes and changelog so downstream users can see what happened.

Never reuse a version number after any part of it has been published publicly.
