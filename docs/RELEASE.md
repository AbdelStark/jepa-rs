# Release Process

## Release Goal

A release is publishable only when the workspace matches the real verification
surface described in [`docs/QUALITY_GATES.md`](./QUALITY_GATES.md).

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
```

Notes:

- Downstream workspace dependencies now carry explicit version requirements for publishable manifests.
- `--exclude-lockfile` is intentional for the downstream crates while the workspace is still unpublished on crates.io. Without it, Cargo attempts to resolve `jepa-core` from the registry when generating a lockfile for the packaged crate.
- After publishing `jepa-core`, maintainers may rerun downstream `cargo package` or `cargo publish --dry-run` commands without `--exclude-lockfile` if they want to confirm registry resolution before publishing the remaining crates.

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

## Support Policy

- Supported status must be stated explicitly in the README and release notes.
- If ONNX loading or parity coverage has limits, those limits must be written
  down for the released version.
- A release is not called production-grade unless the P0 and P1 gaps in
  `PRODUCTION_GAP.md` are closed.

## Release Steps

1. Update version numbers and changelog entries.
2. Run the full verification surface locally, including `scripts/run_parity_suite.sh`.
3. Run the publishability checks in dependency order.
4. Tag the release candidate in git.
5. Publish crates in dependency order.
6. Rebuild docs and verify docs.rs metadata.
7. Announce the release with status, scope, and known limitations.
