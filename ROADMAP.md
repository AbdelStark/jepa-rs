# Roadmap

## Goal

This roadmap defines the sequence for turning `jepa-rs` into a production-grade JEPA framework in Rust without losing correctness to premature feature work.

Use this document together with:

- [PRODUCTION_GAP.md](./PRODUCTION_GAP.md) for the gap register
- [WORK_PACKAGES.md](./WORK_PACKAGES.md) for implementation-sized tasks
- [ARCHITECTURE.md](./ARCHITECTURE.md) for current crate boundaries and invariants

## Roadmap Principles

1. Correctness before convenience.
2. Faithful JEPA semantics before performance optimization.
3. Differential proof before release claims.
4. Typed failures before silent fallbacks.
5. Publish only after the verification surface is strong enough to preserve trust.

## Status Snapshot

As of March 12, 2026:

- M0 is complete.
- M1 is complete: strict image and video masked paths landed with no-leakage regression coverage.
- M2 is complete for the first reference path: the repo now runs fixture-backed strict I-JEPA parity in CI and exposes fallible alternatives for high-risk caller misuse.
- M3 is substantially complete for the current scope: coverage, fuzz, benchmark smoke, parity, and package smoke are part of the enforced verification surface; ONNX remains intentionally scoped to metadata inspection and initializer loading.
- M4 is in progress: local release-candidate hardening is complete, but the first external crates.io release still needs to be exercised.
- M5 remains open.

## Milestones

| Milestone | Outcome | Includes | Exit Criteria | Relative Complexity |
|-----------|---------|----------|---------------|---------------------|
| M0 | Planning baseline | gap register, roadmap, work packages, architecture notes | Planning docs committed and linked from repo entry points | S |
| M1 | Semantic correctness foundation | strict masked image path, strict masked video path, no-leakage tests, trainer contract clarification | Hidden targets cannot influence context encoding in verified image and video paths | XL |
| M2 | Reference validation foundation | fixture-backed differential tests against Python reference, tighter invariants, fallible runtime APIs where needed | CI proves parity for at least one strict image JEPA path and key runtime APIs are safer | L |
| M3 | Quality hardening and scoped interop | fuzzing, benchmark smoke, coverage policy, package smoke, ONNX metadata and initializer loading | Quality gates are materially stronger and ONNX scope is documented precisely | XL |
| M4 | Release candidate | crates.io readiness, package docs, changelog discipline, compatibility policy, examples | Local package smoke passes and public-facing docs support day-one external use | M |
| M5 | Production-grade 1.0 | broader parity coverage, exercised release process, maintained support expectations, any approved interop expansion | Remaining P0 and P1 gaps are closed and the release process has been exercised end to end | L |

## Recommended Execution Order

### Phase 1: Semantic Correctness First

Status: complete

Start here immediately.

Work:

- finalize the design for strict masked encoding
- implement the image training path
- extend the design to the video path
- add regression tests proving no attention leakage from hidden targets

Reason:

Everything else is downstream of semantic correctness. Differential tests and interop work are lower-signal if the training path itself is only approximate.

### Phase 2: Prove Numerical Behavior

Status: complete for one strict image flow

Do this immediately after Phase 1.

Work:

- build a differential harness against `facebookresearch/ijepa`
- define tolerated error bands for key outputs
- tighten runtime validation and panic boundaries in public APIs

Reason:

Once the semantics are correct, the next risk is silent numerical drift.

### Phase 3: Harden The Quality Gates

Status: complete for the current gate set

Do this once parity work is in motion.

Work:

- add fuzz targets for masking, gather, energy, and checkpoint parsing
- define benchmark baselines for masking, core training orchestration, and planning
- add stronger CI policy around coverage and regression checks

Reason:

At that point, the project needs machine-enforced protection against future regressions.

### Phase 4: Complete Interop And Release Engineering

Status: in progress

Do this after the correctness surface is strong.

Work:

- keep ONNX runtime execution out of scope unless it is explicitly approved
- document release packaging and versioning policy
- prepare crates for publication

Reason:

Interop and publishing should not outpace correctness and verification. For the current roadmap, ONNX metadata inspection and initializer loading are enough; full runtime execution is a separate approved scope decision.

## Immediate Next Steps

These are the next concrete steps to execute in order:

1. Broaden parity coverage beyond the bundled tiny strict image fixture so regressions are tested across more shapes and masking layouts.
2. Exercise the first real release candidate flow in dependency order, including changelog and publish dry-runs.
3. Decide whether production-grade scope requires ONNX runtime execution or whether the documented adapter boundary remains acceptable.
4. Tighten operational guidance for external users, especially release support expectations and debugging runbooks.
5. Revisit performance baselines once the first public release shape is fixed.

## Suggested Sprint Structure

### Sprint 1

- expand WP-004 parity coverage
- rehearse the first publish flow described in WP-008

### Sprint 2

- close remaining operational-documentation and support-policy gaps
- refresh benchmark baselines once release inputs stabilize

### Sprint 3

- execute WP-007 only if ONNX runtime expansion is explicitly approved
- convert release-candidate hardening into the first external release

## Milestone Exit Tests

### M1 exit tests

- image path has explicit no-leakage regression coverage
- video path has explicit no-leakage regression coverage
- generic trainer docs clearly state where approximation remains and where strict paths should be used

### M2 exit tests

- differential suite runs locally with one command
- parity checks cover at least one image encoder-predictor-energy flow
- public runtime misuse no longer surfaces as undocumented panics in the highest-risk APIs

### M3 exit tests

- fuzzing exists for masking and energy code
- benchmarks have checked-in baselines and review policy
- CI blocks regressions across the expanded quality surface

### M4 exit tests

- crate metadata is complete
- public docs include versioned status, support scope, and limitations
- release process is documented and local package smoke passes in dependency order

### M5 exit tests

- readiness checklist is complete
- remaining P0 and P1 gaps are closed
- maintainers are comfortable attaching the “production-grade” label publicly

## Work Deferred Until After M3

- distributed training
- new modality support beyond the current scope
- large model zoo distribution
- advanced serving integrations

These are good follow-on investments, but not the shortest path to trustworthy production quality.
