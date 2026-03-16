# Release Process

## Current Policy

As of 2026-03-16, releases should be treated as alpha-quality library and local
tooling drops. There is no fully automated publish pipeline yet.

## Preconditions

Before cutting a release candidate:

- `CHANGELOG.md` reflects all user-visible changes.
- `README.md`, `CONTRIBUTING.md`, and `CLAUDE.md` describe the current repo.
- The commands in [QUALITY_GATES.md](./QUALITY_GATES.md) pass.
- Known limitations remain truthful and visible.

## Candidate Checklist

1. Run the release-candidate gate from [QUALITY_GATES.md](./QUALITY_GATES.md).
2. Re-read the README status and limitations sections for truthfulness.
3. Confirm any browser-demo changes still describe the CPU-backed exported path
   honestly unless runtime WebGPU selection was validated in the same release.
4. Confirm ONNX graph execution is not described as production-grade unless the
   runtime path was explicitly hardened and re-verified.
5. Update or confirm the milestone and blocker docs:
   - [ROADMAP.md](./ROADMAP.md)
   - [PRODUCTION_GAPS.md](./PRODUCTION_GAPS.md)

## Publish Order

The workspace crates depend on each other by version, so first-time publication
must respect dependency order.

Safe base order from current manifests:

1. `jepa-core`
2. `jepa-vision`
3. `jepa-world`
4. `jepa-train`
5. `jepa-compat`
6. `jepa-web` (manual judgment; not in current package smoke)
7. `jepa` (manual judgment; not in current package smoke)

`jepa-web` and `jepa` should not be published casually until their package and
install expectations are covered by repeatable smoke checks.

## Tagging And Follow-Up

- Tag the release only after the publishable crates succeed.
- Verify docs.rs builds for published library crates.
- Confirm the GitHub Actions CI run for the release commit is green.
- If a published crate is broken, yank it and cut a follow-up patch instead of
  rewriting history.
