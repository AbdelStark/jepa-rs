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
- `cargo llvm-cov --workspace --all-features --fail-under-lines 80`
- Fuzz smoke runs complete for all targets
- Benchmarks compile, and any significant regression is reviewed
- README, changelog, and crate metadata reflect current capabilities

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
2. Run the full verification surface locally.
3. Tag the release candidate in git.
4. Publish crates in dependency order.
5. Rebuild docs and verify docs.rs metadata.
6. Announce the release with status, scope, and known limitations.
