# Production Gap Analysis

## Purpose

This document is the working gap register for taking `jepa-rs` from its current alpha state to a production-grade JEPA framework in Rust.

As of March 13, 2026, the workspace is useful for local experimentation and extension work, and the release-candidate verification surface is materially stronger than it was at the start of this cycle. It is still not production-grade yet.

## Target Bar

`jepa-rs` reaches the target bar when all of the following are true:

1. The training path enforces faithful JEPA masking semantics before encoder attention.
2. Core numerical behavior is verified against reference Python implementations.
3. Public APIs fail predictably with typed errors instead of surprising panics for runtime misuse.
4. Interop paths are functional for both safetensors and ONNX.
5. CI enforces correctness, linting, formatting, documentation, fuzzing, and benchmark regression checks.
6. The crates are publishable, versioned, and documented for external users.

## Current Declaration

Current status: **alpha, release-candidate rehearsal complete locally**

Current strengths:

- workspace builds cleanly
- unit, integration, property, doc, clippy, and rustdoc checks pass locally
- safetensors and ONNX checkpoint inspection/loading are usable
- strict image and video masked forward paths have regression coverage
- strict I-JEPA image parity runs against three checked-in canonical Python fixtures in CI
- caller-triggerable misuse has typed or fallible paths in the highest-risk vision, training, and world-model APIs
- coverage, fuzzing, benchmark smoke, and release-policy performance budgets are part of the verification surface
- dependency-order package smoke passes for all crates with the documented unpublished-workspace packaging procedure
- external-facing runbooks now cover parity triage, package smoke, release rollback, and support boundaries
- core crate boundaries are already reasonably clear

Current blockers:

- parity coverage remains image-only; strict video parity is still unproven
- the first crates.io release still needs approval and actual publication

## Gap Register

| ID | Gap | Severity | Status | Why It Matters | Affected Areas | Target State |
|----|-----|----------|--------|----------------|----------------|--------------|
| G-001 | Generic JEPA orchestration remains approximate | P0 | Closed for current scope | Hidden tokens can still influence encoder attention in the generic training helper, but strict image and video paths now exist and are the documented semantic reference | `jepa-train`, `jepa-vision` | Strict modality-specific paths exist and the generic helper is clearly documented as approximate |
| G-002 | Reference parity is not proven | P0 | Closed for bundled strict image fixture set | A large local suite is not enough for numerical ML code; without differential tests, silent drift can ship | `jepa-core`, `jepa-vision`, `jepa-train`, `specs` | Differential tests run against a maintained canonical Python JEPA fixture set in CI |
| G-003 | Runtime validation is inconsistent across public APIs | P1 | Closed for highest-risk caller paths | Panic-based behavior is acceptable for invariant violations, not for ordinary runtime misuse by library callers | `jepa-world`, `jepa-vision`, `jepa-train` | Fallible or clearly documented APIs exist for user-triggerable failure modes |
| G-004 | ONNX runtime execution is narrower than the adapter surface | P1 | Closed for current scope | Metadata and initializer loading now work, and the supported boundary explicitly excludes graph execution unless maintainers approve a separate expansion | `jepa-compat` | ONNX model info extraction and weight loading work against real models, with runtime scope documented explicitly |
| G-005 | Quality gates need continuous enforcement discipline | P1 | Closed for the current gate set | Coverage, fuzz, benchmark smoke, parity, and package smoke only help if they stay release blockers | workspace, `.github` | CI enforces a stronger verification surface including parity, fuzz, coverage, package smoke, and performance checks |
| G-006 | Release readiness is incomplete | P1 | In progress | External users still lack a first crates.io release, exercised migration discipline, and a published support story | workspace docs, manifests, release process | Crates are publishable with clear versioning, changelog discipline, and contributor guidance |
| G-007 | Operational and debugging guidance is thin | P2 | Closed for current scope | New contributors and downstream users should not have to infer routine verification or release steps from source code | README, docs | Runbooks, limitations, and support expectations are documented explicitly |
| G-008 | Performance validation is mostly anecdotal | P2 | Closed for current release policy | The code may be fast enough, but baselines and regression policy still need continued maintenance | `benches`, docs | Core training, masking, strict image flow, and planning paths have documented benchmark baselines and regression review thresholds |

## Dependency Graph

The remaining work is not independent. The correct order from here is:

1. finish G-006 by publishing the first public release only after approval
2. expand parity beyond strict image flows only if maintainers decide strict video parity is a pre-`1.0` requirement
3. keep the operational runbooks and performance budgets current as the release surface changes
4. execute G-004 only if ONNX runtime expansion is explicitly approved

## Immediate Focus

The next highest-value sequence is:

1. Publish the first crates.io release only after maintainers approve the rehearsed flow and versioning inputs.
2. Decide whether strict video parity is required before the project claims production-grade readiness.
3. Maintain the runbooks, release notes, and benchmark budgets as the public surface changes.
4. Only pull ONNX runtime execution into scope if maintainers explicitly decide that production-grade requires it.

## Out Of Scope Until After Semantic Correctness

The following work should not distract from the remaining open gaps:

- distributed training
- model zoo packaging
- large-scale performance tuning
- speculative new RFCs beyond the existing specification

## Exit Condition For “Production-Grade”

Do not call `jepa-rs` production-grade until the remaining P0 and P1 gaps are closed, the release readiness milestone in [ROADMAP.md](./ROADMAP.md) is exercised end to end, and the supported ONNX scope is either accepted as-is or expanded by explicit approval.
