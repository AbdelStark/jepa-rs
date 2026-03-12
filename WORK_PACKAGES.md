# Work Packages

This file decomposes the roadmap into implementation-sized packages that can be delegated directly.

Read first:

- [PRODUCTION_GAP.md](./PRODUCTION_GAP.md)
- [ROADMAP.md](./ROADMAP.md)
- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [SPECIFICATION.md](./SPECIFICATION.md)

## WP-001: Strict Masked Encoder Design

Status: complete

**Objective**: Decide how strict pre-encoder masking will work for image and video paths without casually breaking public contracts.

**Why**: The current generic trainer slices tokens after encoder forward. That is an approximation, not faithful JEPA semantics.

**Context**:
- Read RFC-002, RFC-003, RFC-005, RFC-008
- Inspect `crates/jepa-train/src/trainer.rs`
- Inspect `crates/jepa-vision/src/vit.rs`
- Inspect `crates/jepa-vision/src/video.rs`

**Deliverables**:
- A short design note or ADR committed to the repo
- Clear decision on whether the strict path is:
  - encoder-specific helper methods
  - a new internal training trait
  - or a small approved public API extension
- Explicit migration path from the current generic helper

**Acceptance criteria**:
- [ ] Design preserves correctness as the primary goal
- [ ] Public API changes, if any, are explicitly escalated before implementation
- [ ] The design defines how image and video share the pattern

**Dependencies**: none

## WP-002: Image Strict Masked Path

Status: complete

**Objective**: Implement a strict masked training path for image JEPA where hidden target patches do not influence context encoder attention.

**Context**:
- Read RFC-002, RFC-003, RFC-005, RFC-008
- Modify `crates/jepa-vision/src/vit.rs`
- Modify `crates/jepa-vision/src/image.rs`
- Modify `crates/jepa-train/src/trainer.rs` only if the chosen design requires it

**Deliverables**:
- Masked image encoder or training helper
- Regression tests proving no leakage from hidden patches
- Clear inline docs on the strict path

**Acceptance criteria**:
- [ ] Hidden target patches cannot influence context encoding in tests
- [ ] Existing tests still pass
- [ ] New tests fail against the old behavior and pass against the new one

**Dependencies**: WP-001

## WP-003: Video Strict Masked Path

Status: complete

**Objective**: Extend the strict masked design to the video JEPA path.

**Context**:
- Read RFC-002, RFC-003, RFC-005, RFC-008
- Modify `crates/jepa-vision/src/video.rs`
- Modify `crates/jepa-train/src/trainer.rs` only if required by the chosen design

**Deliverables**:
- Strict masked video training path
- No-leakage tests for spatiotemporal masking
- Clear docs on any modality-specific constraints

**Acceptance criteria**:
- [ ] Hidden target tubelets cannot influence context encoding in tests
- [ ] Existing video tests still pass
- [ ] Image and video strict paths are conceptually consistent

**Dependencies**: WP-001, WP-002

## WP-004: Differential Parity Harness

Status: complete for the bundled strict image fixture set, with strict video parity still remaining

**Objective**: Prove that key image-path behavior matches a Python reference implementation closely enough to trust refactors.

**Context**:
- Read RFC-002, RFC-003, RFC-004, RFC-005, RFC-008
- Read `README.md` reference implementations section
- Add tests under `specs` or crate integration tests

**Deliverables**:
- Local differential test harness
- Fixture or adapter instructions for `facebookresearch/ijepa`
- Documented numeric tolerances

**Acceptance criteria**:
- [ ] One command runs the parity suite locally
- [ ] At least one end-to-end image JEPA flow is compared against the reference implementation
- [ ] CI integration path is defined, even if initially optional

**Dependencies**: WP-002

## WP-005: Runtime Validation Cleanup

Status: complete for the highest-risk caller-triggerable misuse paths

**Objective**: Reduce surprising runtime panics in public-facing APIs and replace them with clearer contracts or fallible alternatives.

**Context**:
- Inspect `crates/jepa-world/src/planner.rs`
- Inspect `crates/jepa-world/src/memory.rs`
- Inspect other public modules for caller-triggerable panics

**Deliverables**:
- Panic audit
- `try_*` APIs or typed errors where needed
- Tests for documented failure modes

**Acceptance criteria**:
- [ ] High-risk runtime misuse has a typed failure path
- [ ] Remaining panics are true invariant checks and documented as such
- [ ] Public docs name the failure behavior explicitly

**Dependencies**: none

## WP-006: Fuzzing, Coverage, And Benchmark Hardening

Status: complete for the current verification surface, with baseline maintenance still ongoing

**Objective**: Strengthen machine-enforced quality gates beyond the current unit and property suite.

**Context**:
- Add fuzz targets for masking, gather, energy, and checkpoint parsing
- Extend benchmark coverage for trainer and planner hot paths
- Update CI if new commands are added

**Deliverables**:
- Fuzz targets and runner instructions
- Benchmark baselines and regression policy
- Proposed coverage policy

**Acceptance criteria**:
- [ ] Fuzzing exists for the highest-risk unbounded-input code
- [ ] Benchmarks exist for core training and planning paths
- [ ] CI integration plan is documented

**Dependencies**: WP-002, WP-003, WP-004

## WP-007: ONNX Runtime Integration

Status: deferred pending explicit scope and dependency approval

**Objective**: Expand the current parser-backed ONNX adapter to functional runtime-backed execution if that scope is explicitly approved.

**Context**:
- Modify `crates/jepa-compat/src/onnx.rs`
- Add any needed compatibility tests
- This package requires dependency approval before implementation

**Deliverables**:
- Runtime-backed model info extraction
- Real ONNX parsing tests
- Clear error mapping for unsupported models

**Acceptance criteria**:
- [ ] Real ONNX files can be inspected
- [ ] Missing runtime and invalid model states are clearly distinguished
- [ ] The README and docs describe the supported runtime boundary precisely

**Dependencies**: human approval for dependency changes

## WP-008: Release Candidate Hardening

Status: complete for local release-candidate rehearsal; actual external publish still pending

**Objective**: Prepare the workspace for external publication and long-lived maintenance.

**Context**:
- Update README, CHANGELOG, CONTRIBUTING, and crate metadata as needed
- Add release process notes

**Deliverables**:
- crates.io readiness checklist
- compatibility and support policy
- polished package docs and examples

**Acceptance criteria**:
- [ ] Package metadata is complete and accurate
- [ ] User-facing docs match real capabilities and limitations
- [ ] Release process is documented and repeatable

**Dependencies**: WP-004, WP-006, WP-007

## Current Recommended Start Order

1. Publish the first crates.io release only after maintainers explicitly approve the rehearsed dependency-order flow.
2. Expand differential coverage further only if maintainers decide strict video parity is required before `1.0`.
3. Keep the benchmark budgets and operator runbooks current as the release surface changes.
4. Execute WP-007 only if ONNX runtime expansion is explicitly approved.
