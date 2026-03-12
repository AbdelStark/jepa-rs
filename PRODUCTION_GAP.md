# Production Gap Analysis

## Purpose

This document is the working gap register for taking `jepa-rs` from its current alpha state to a production-grade JEPA framework in Rust.

As of March 12, 2026, the workspace is useful for local experimentation and extension work, and the release-candidate verification surface is materially stronger than it was at the start of this cycle. It is still not production-grade yet.

## Target Bar

`jepa-rs` reaches the target bar when all of the following are true:

1. The training path enforces faithful JEPA masking semantics before encoder attention.
2. Core numerical behavior is verified against reference Python implementations.
3. Public APIs fail predictably with typed errors instead of surprising panics for runtime misuse.
4. Interop paths are functional for both safetensors and ONNX.
5. CI enforces correctness, linting, formatting, documentation, fuzzing, and benchmark regression checks.
6. The crates are publishable, versioned, and documented for external users.

## Current Declaration

Current status: **alpha, release-candidate hardening complete locally**

Current strengths:

- workspace builds cleanly
- unit, integration, property, doc, clippy, and rustdoc checks pass locally
- safetensors and ONNX checkpoint inspection/loading are usable
- strict image and video masked forward paths have regression coverage
- strict I-JEPA image parity runs against a checked-in canonical Python fixture in CI
- caller-triggerable misuse has typed or fallible paths in the highest-risk vision, training, and world-model APIs
- coverage, fuzzing, and benchmark smoke checks are part of the verification surface
- package smoke passes for all crates with the documented unpublished-workspace packaging procedure
- core crate boundaries are already reasonably clear

Current blockers:

- parity coverage is still narrow and currently proves only one strict image flow
- the first crates.io release has not been exercised yet
- contributor-facing operational guidance still needs more explicit runbooks and support boundaries
- full ONNX runtime execution is still out of scope

## Gap Register

| ID | Gap | Severity | Status | Why It Matters | Affected Areas | Target State |
|----|-----|----------|--------|----------------|----------------|--------------|
| G-001 | Generic JEPA orchestration remains approximate | P0 | Closed for current scope | Hidden tokens can still influence encoder attention in the generic training helper, but strict image and video paths now exist and are the documented semantic reference | `jepa-train`, `jepa-vision` | Strict modality-specific paths exist and the generic helper is clearly documented as approximate |
| G-002 | Reference parity is not proven | P0 | Closed for first reference path | A large local suite is not enough for numerical ML code; without differential tests, silent drift can ship | `jepa-core`, `jepa-vision`, `jepa-train`, `specs` | Differential tests run against at least one canonical Python JEPA implementation in CI |
| G-003 | Runtime validation is inconsistent across public APIs | P1 | Closed for highest-risk caller paths | Panic-based behavior is acceptable for invariant violations, not for ordinary runtime misuse by library callers | `jepa-world`, `jepa-vision`, `jepa-train` | Fallible or clearly documented APIs exist for user-triggerable failure modes |
| G-004 | ONNX runtime execution is narrower than the adapter surface | P1 | Open | Metadata and initializer loading now work, but ONNX runtime execution is still outside the supported scope | `jepa-compat` | ONNX model info extraction and weight loading work against real models, with runtime scope documented explicitly |
| G-005 | Quality gates need continuous enforcement discipline | P1 | Closed for the current gate set | Coverage, fuzz, benchmark smoke, parity, and package smoke only help if they stay release blockers | workspace, `.github` | CI enforces a stronger verification surface including parity, fuzz, coverage, package smoke, and performance checks |
| G-006 | Release readiness is incomplete | P1 | In progress | External users still lack a first crates.io release, exercised migration discipline, and a published support story | workspace docs, manifests, release process | Crates are publishable with clear versioning, changelog discipline, and contributor guidance |
| G-007 | Operational and debugging guidance is thin | P2 | In progress | New contributors and downstream users still need to infer too much from source code | README, architecture docs, examples | Runbooks, examples, limitations, and support expectations are documented explicitly |
| G-008 | Performance validation is mostly anecdotal | P2 | In progress | The code may be fast enough, but baselines and regression policy still need continued maintenance | `benches`, CI | Core training, masking, and planning paths have benchmark baselines and regression detection |

## Dependency Graph

The remaining work is not independent. The correct order from here is:

1. broaden G-002 parity coverage beyond the first strict image fixture
2. finish G-006 by exercising the first public release flow
3. close G-007 with explicit operational runbooks and support guidance
4. continue G-008 by maintaining benchmark baselines and regression budgets
5. execute G-004 only if ONNX runtime expansion is explicitly approved

## Immediate Focus

The next highest-value sequence is:

1. Add at least one more canonical parity fixture so numerical coverage is not concentrated in a single tiny image case.
2. Rehearse the first crates.io release candidate in dependency order using the documented package smoke and release steps.
3. Expand contributor-facing operational docs where external users would currently have to infer too much.
4. Only pull ONNX runtime execution into scope if maintainers explicitly decide that production-grade requires it.

## Out Of Scope Until After Semantic Correctness

The following work should not distract from the remaining open gaps:

- distributed training
- model zoo packaging
- large-scale performance tuning
- speculative new RFCs beyond the existing specification

## Exit Condition For “Production-Grade”

Do not call `jepa-rs` production-grade until the remaining P0 and P1 gaps are closed, the release readiness milestone in [ROADMAP.md](./ROADMAP.md) is exercised end to end, and the supported ONNX scope is either accepted as-is or expanded by explicit approval.
