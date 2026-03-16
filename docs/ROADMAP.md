# Roadmap

## Current State

As of 2026-03-16, `jepa-rs` is alpha. It is suitable for research, local demos,
and library experimentation. It is not yet suitable for claiming unqualified
production readiness across every surface.

## Milestone 1: Release-Candidate Truthfulness

Goal: make the repo safe to consume as an alpha library and local tooling stack.

Exit criteria:

- Repo docs and agent context match the actual 7-crate workspace.
- Verification and release runbooks exist and are accurate.
- Browser-demo and ONNX limitations are stated honestly in public docs.
- Workspace gates pass cleanly on a fresh checkout.

Included:

- README, CONTRIBUTING, CHANGELOG, `CLAUDE.md`, and runbook upkeep
- Caller-facing validation and tests for boundary-heavy demo surfaces

Deferred:

- New model features
- Manifest or CI redesign

Complexity: M

## Milestone 2: Strict Video And Runtime Confidence

Goal: reduce the highest remaining silent-regression risks in non-image paths.

Exit criteria:

- Fixture-driven strict video parity exists with CI coverage.
- ONNX runtime behavior has explicit supported-shape documentation and regression tests.
- Runtime mismatch errors stay typed and actionable.

Included:

- V-JEPA parity fixtures and tests
- ONNX runtime hardening
- Documentation updates for supported and unsupported runtime cases

Deferred:

- Full browser GPU deployment
- New training recipes beyond current scope

Dependencies:

- Stable fixture policy
- Agreement on acceptable parity tolerances

Complexity: L

## Milestone 3: Browser Demo Backend Selection

Goal: either ship a validated WebGPU path or keep the browser demo explicitly CPU-only.

Exit criteria:

- Runtime backend selection is implemented and tested, or the CPU-only path is
  intentionally locked in with matching UI and docs.
- At least one browser execution path is documented end to end.
- Failure modes for unsupported browser environments are explicit.

Included:

- `jepa-web` runtime-selection work
- Browser-facing docs and UX copy
- Verification notes for local demo execution

Deferred:

- Hosted production deployment guarantees
- Large-model browser training claims

Dependencies:

- Stable WASM build workflow
- Clear support target for browsers and backends

Complexity: L
