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

## Milestones

| Milestone | Outcome | Includes | Exit Criteria | Relative Complexity |
|-----------|---------|----------|---------------|---------------------|
| M0 | Planning baseline | gap register, roadmap, work packages, architecture notes | Planning docs committed and linked from repo entry points | S |
| M1 | Semantic correctness foundation | strict masked image path, strict masked video path, no-leakage tests, trainer contract clarification | Hidden targets cannot influence context encoding in verified image and video paths | XL |
| M2 | Reference validation foundation | differential tests against Python reference, tighter invariants, fallible runtime APIs where needed | CI proves parity for at least one image JEPA path and key runtime APIs are safer | L |
| M3 | Hardening and interop | fuzzing, benchmark baselines, coverage policy, ONNX runtime integration | Quality gates are materially stronger and ONNX is no longer a stub | XL |
| M4 | Release candidate | crates.io readiness, package docs, changelog discipline, compatibility policy, examples | Crates are publishable and public-facing docs support day-one external use | M |
| M5 | Production-grade 1.0 | stable API story, verified release process, maintained support expectations | P0 and P1 gaps closed, release process exercised, readiness checklist complete | L |

## Recommended Execution Order

### Phase 1: Semantic Correctness First

Start here immediately.

Work:

- finalize the design for strict masked encoding
- implement the image training path
- extend the design to the video path
- add regression tests proving no attention leakage from hidden targets

Reason:

Everything else is downstream of semantic correctness. Differential tests and interop work are lower-signal if the training path itself is only approximate.

### Phase 2: Prove Numerical Behavior

Do this immediately after Phase 1.

Work:

- build a differential harness against `facebookresearch/ijepa`
- define tolerated error bands for key outputs
- tighten runtime validation and panic boundaries in public APIs

Reason:

Once the semantics are correct, the next risk is silent numerical drift.

### Phase 3: Harden The Quality Gates

Do this once parity work is in motion.

Work:

- add fuzz targets for masking, gather, energy, and checkpoint parsing
- define benchmark baselines for masking, core training orchestration, and planning
- add stronger CI policy around coverage and regression checks

Reason:

At that point, the project needs machine-enforced protection against future regressions.

### Phase 4: Complete Interop And Release Engineering

Do this after the correctness surface is strong.

Work:

- implement ONNX runtime support
- document release packaging and versioning policy
- prepare crates for publication

Reason:

Interop and publishing should not outpace correctness and verification.

## Immediate Next Steps

These are the next concrete steps to execute in order:

1. Create a small design note or ADR for strict masked encoder semantics.
2. Add failing image-path tests that prove hidden patches cannot affect context encoding.
3. Implement the image masked path with minimal public API churn.
4. Repeat the same pattern for video.
5. Add a first differential suite against `facebookresearch/ijepa`.
6. Audit panic-based public APIs and add fallible alternatives where the failure mode is caller-triggerable.

## Suggested Sprint Structure

### Sprint 1

- WP-001 strict masked encoder design
- WP-002 image strict masked path

### Sprint 2

- WP-003 video strict masked path
- WP-004 image differential harness

### Sprint 3

- WP-005 runtime validation cleanup
- WP-006 fuzz and benchmark hardening

### Sprint 4

- WP-007 ONNX runtime integration
- WP-008 release candidate hardening

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
- release process is documented and rehearsed

### M5 exit tests

- readiness checklist is complete
- P0 and P1 gaps are closed
- maintainers are comfortable attaching the “production-grade” label publicly

## Work Deferred Until After M3

- distributed training
- new modality support beyond the current scope
- large model zoo distribution
- advanced serving integrations

These are good follow-on investments, but not the shortest path to trustworthy production quality.
