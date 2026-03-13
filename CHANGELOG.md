# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **jepa-core**: `Representation::gather` now uses indexed selection and preserves representation masks instead of dropping them silently
- **jepa-train**: `JepaComponents::forward_step` now validates generated masks and passes real target indices to predictors
- **jepa-train**: Generic `JepaComponents::forward_step` docs now explicitly call out its approximate masking semantics and point callers to strict vision helpers
- **jepa-vision**: `TransformerPredictor` now conditions prediction tokens on target positions instead of ignoring them
- **jepa-compat**: ONNX support now parses real `ModelProto` files and loads initializers into the checkpoint abstraction
- `scripts/run_parity_suite.sh` now runs every checked-in strict image fixture by default and can target a single fixture or fixture directory explicitly
- Documentation and agent context files now describe the project as alpha and explicitly call out the current trainer and ONNX limitations
- Added in-repo planning docs for production gaps, milestone roadmap, and implementation work packages

### Added
- **jepa**: CLI binary with 6 subcommands (`models`, `inspect`, `checkpoint`, `train`, `encode`, `tui`)
- **jepa**: Interactive TUI dashboard with 5 tabs (Dashboard, Models, Training, Checkpoint, About) using ratatui and Catppuccin Mocha theme
- **jepa-vision**: Strict masked image and video forward paths with no-leakage regression coverage
- **jepa-vision**: Criterion coverage for strict `IJepa::forward_step_strict` in the maintained vision benchmark surface
- **jepa-world**: `try_new`, `try_push`, `try_total_cost`, and `try_plan` runtime-validation helpers for caller-triggerable failure modes
- **jepa-compat**: Parser-backed ONNX metadata inspection and initializer loading
- Two additional canonical Python-exported strict I-JEPA image fixtures covering non-square grids and RGB asymmetric patches
- External-facing operations, release-candidate, and performance-budget runbooks under `docs/`
- Fuzz targets for masking, gather, energy, and checkpoint parsing
- Coverage, fuzz, and benchmark-smoke CI jobs
- ADR-0001 for strict masked encoder semantics
- Quality-gate and release-process runbooks under `docs/`
- **jepa-core**: Core traits (`Encoder`, `Predictor`, `EnergyFn`, `MaskingStrategy`, `CollapseRegularizer`) — RFC-001 through RFC-007
- **jepa-core**: Energy functions: `L2Energy`, `CosineEnergy`, `SmoothL1Energy` — RFC-004
- **jepa-core**: Masking strategies: `BlockMasking`, `SpatiotemporalMasking`, `MultiBlockMasking` — RFC-005
- **jepa-core**: Collapse prevention: `VICReg`, `BarlowTwins` — RFC-006
- **jepa-core**: EMA with `CosineMomentumSchedule` for target encoder updates — RFC-007
- **jepa-core**: `JepaConfig` builder with validation and presets
- **jepa-vision**: `VitEncoder` with `TransformerBlock`, `MHSA`, `MLP` — RFC-002
- **jepa-vision**: `PatchEmbedding` for image patchification — RFC-002
- **jepa-vision**: `RotaryPositionEncoding2D` for spatial awareness — RFC-002
- **jepa-vision**: `TransformerPredictor` and `IJepa` model — RFC-003
- **jepa-vision**: `VitVideoEncoder`, `TubeletEmbedding`, 3D RoPE, `VJepa` — RFC-003
- **jepa-world**: `Action`, `ActionConditionedPredictor` trait — RFC-009
- **jepa-world**: `WorldModel`, `RandomShootingPlanner` (CEM), `L2Cost` — RFC-009
- **jepa-world**: `HierarchicalJepa` (H-JEPA) — RFC-010
- **jepa-world**: `ShortTermMemory` ring buffer — RFC-010
- **jepa-train**: `JepaComponents` forward step orchestration — RFC-008
- **jepa-train**: `WarmupCosineSchedule` learning rate scheduler — RFC-008
- **jepa-train**: `CheckpointMeta` serialization — RFC-008
- **jepa-train**: `TrainConfig`, `TrainMetrics`, `TrainStepOutput` — RFC-008
- **jepa-compat**: Safetensors checkpoint loading and conversion — RFC-011
- **jepa-compat**: I-JEPA/V-JEPA key mapping patterns — RFC-011
- CI workflow with check, test, clippy, fmt, and doc jobs
- 365 unit/integration tests + doc tests across all crates
- Property-based tests with proptest for numerical invariants
- Criterion benchmarks for core and vision crates
- 3 runnable examples (I-JEPA demo, training loop, world model planning)
