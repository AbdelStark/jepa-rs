# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- **jepa-compat**: ONNX model info types (API-complete, runtime stub) — RFC-011
- CI workflow with check, test, clippy, fmt, and doc jobs
- 245 unit/integration tests + 16 doc tests across all crates
- Property-based tests with proptest for numerical invariants
- Criterion benchmarks for core and vision crates
- 3 runnable examples (I-JEPA demo, training loop, world model planning)
