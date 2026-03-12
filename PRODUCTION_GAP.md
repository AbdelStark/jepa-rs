# Production Gap Analysis

## Purpose

This document is the working gap register for taking `jepa-rs` from its current alpha state to a production-grade JEPA framework in Rust.

As of March 12, 2026, the workspace is useful for local experimentation and extension work, but it is not yet suitable for parity-sensitive production training or deployment.

## Target Bar

`jepa-rs` reaches the target bar when all of the following are true:

1. The training path enforces faithful JEPA masking semantics before encoder attention.
2. Core numerical behavior is verified against reference Python implementations.
3. Public APIs fail predictably with typed errors instead of surprising panics for runtime misuse.
4. Interop paths are functional for both safetensors and ONNX.
5. CI enforces correctness, linting, formatting, documentation, fuzzing, and benchmark regression checks.
6. The crates are publishable, versioned, and documented for external users.

## Current Declaration

Current status: **alpha**

Current strengths:

- workspace builds cleanly
- unit, integration, property, doc, clippy, and rustdoc checks pass locally
- safetensors and ONNX checkpoint inspection/loading are usable
- strict image and video masked forward paths have regression coverage
- coverage, fuzzing, and benchmark smoke checks are part of the verification surface
- core crate boundaries are already reasonably clear

Current blockers:

- differential parity is unproven
- release engineering and public package readiness are incomplete
- reference parity is still fixture-driven rather than mandatory in CI
- full ONNX runtime execution is still out of scope

## Gap Register

| ID | Gap | Severity | Why It Matters | Affected Areas | Target State |
|----|-----|----------|----------------|----------------|--------------|
| G-001 | Generic JEPA orchestration remains approximate | P0 | Hidden tokens can still influence encoder attention in the generic training helper, even though strict image and video paths now exist | `jepa-train`, `jepa-vision` | Strict modality-specific paths exist and the generic helper is clearly documented as approximate |
| G-002 | Reference parity is not proven | P0 | A large local suite is not enough for numerical ML code; without differential tests, silent drift can ship | `jepa-core`, `jepa-vision`, `jepa-train`, `specs` | Differential tests run against at least one canonical Python JEPA implementation in CI |
| G-003 | Runtime validation is inconsistent across public APIs | P1 | Panic-based behavior is acceptable for invariant violations, not for ordinary runtime misuse by library callers | `jepa-world`, `jepa-vision`, `jepa-train` | Fallible or clearly documented APIs exist for user-triggerable failure modes |
| G-004 | ONNX runtime execution is narrower than the adapter surface | P1 | Metadata and initializer loading now work, but ONNX runtime execution is still outside the supported scope | `jepa-compat` | ONNX model info extraction and weight loading work against real models, with runtime scope documented explicitly |
| G-005 | Quality gates need continuous enforcement discipline | P1 | Coverage, fuzz, and benchmark checks exist now, but they only help if maintained as release blockers | workspace, `.github` | CI enforces a stronger verification surface including fuzz and performance checks |
| G-006 | Release readiness is incomplete | P1 | External users still lack crates.io releases, API-level stability messaging, compatibility guarantees, and migration discipline | workspace docs, manifests, release process | Crates are publishable with clear versioning, changelog discipline, and contributor guidance |
| G-007 | Operational and debugging guidance is thin | P2 | New contributors and downstream users still need to infer too much from source code | README, architecture docs, examples | Runbooks, examples, limitations, and support expectations are documented explicitly |
| G-008 | Performance validation is mostly anecdotal | P2 | The code may be fast enough, but there are no budgets or regression policies for hot paths | `benches`, CI | Core training, masking, and planning paths have benchmark baselines and regression detection |

## Dependency Graph

The work is not independent. The correct order is:

1. G-001 strict masked training semantics
2. G-002 differential parity on top of the corrected semantics
3. G-003 runtime validation cleanup alongside G-002
4. G-005 quality gates once parity and fuzz targets exist
5. G-004 ONNX scope expansion in parallel once a stronger runtime story is justified
6. G-006 release readiness after the library is semantically correct and verified
7. G-007 and G-008 continuously, but finalized near release candidate

## Immediate Focus

The next highest-value sequence is:

1. Design the masked-encoder training path that preserves current crate boundaries as much as possible.
2. Add failing regression tests proving hidden tokens cannot affect context encoding.
3. Implement the image path first, because it is simpler and should become the reference implementation for video.
4. Extend the same semantics to video.
5. Only after semantic correctness is fixed, wire differential tests against `facebookresearch/ijepa`.

## Out Of Scope Until After Semantic Correctness

The following work should not distract from G-001 and G-002:

- distributed training
- model zoo packaging
- large-scale performance tuning
- speculative new RFCs beyond the existing specification

## Exit Condition For “Production-Grade”

Do not call `jepa-rs` production-grade until all P0 and P1 gaps are closed and the release readiness milestone in [ROADMAP.md](./ROADMAP.md) is complete.
