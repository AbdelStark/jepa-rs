# Project Decisions

Living document for durable architectural and workflow decisions.

Entry format:
- `YYYY-MM-DD`: decision — rationale — alternatives or non-goals

Entries:
- `2026-03-13`: Keep `IJepa::forward_step_strict` and `VJepa::forward_step_strict` as the semantic reference paths — `Encoder::Input` is opaque, so the generic trainer cannot pre-mask tokens faithfully — do not treat `jepa_train::JepaComponents::forward_step` as the strict reference implementation.
- `2026-03-13`: Keep checkpoint and ONNX logic inside `jepa-compat` — format-specific code stays isolated from core model crates and the CLI remains a thin consumer — avoid spreading loading and runtime concerns across model crates.
- `2026-03-13`: Use typed `thiserror` errors in library crates and reserve `anyhow` for CLI and TUI boundaries — callers need structured failure modes in library code, while user-facing commands need contextual error reporting — do not collapse library errors into opaque CLI-style errors.
- `2026-03-13`: Keep strict parity fixture-driven — Python export tooling is optional maintenance support, not a build dependency — do not make the reference Python environment a requirement for normal Rust verification.
- `2026-03-14`: C-JEPA is additive and configuration-based — existing I-JEPA/V-JEPA pipelines are untouched; C-JEPA uses separate structs (`ObjectMasking`, `SlotAttention`, `CausalJepaComponents`, `ObjectDynamicsPredictor`) that users opt into — do not modify existing JEPA behavior.
- `2026-03-14`: `ObjectMasking` lives in `jepa-core` alongside `BlockMasking` and `SpatiotemporalMasking` — object-level masking is a masking strategy variant, not a world-model-only concern — keeps the trait ecosystem consistent.
- `2026-03-14`: Slot attention lives in `jepa-vision` as a module, not a standalone crate — avoids workspace member churn while keeping the component modular — can be extracted later if reuse demand grows.
- `2026-03-14`: C-JEPA training uses a frozen encoder paradigm (no EMA) — this is a separate `CausalJepaComponents` in `jepa-train`, not a mode flag on the existing `JepaComponents` — keeps the two training loops independent and composable.
- `2026-03-16`: Keep the exported `jepa-web` demo on the CPU-backed WASM path until runtime WebGPU selection is validated end to end — deterministic tests and honest browser behavior are more valuable than claiming GPU support prematurely — keep `burn-wgpu` scaffolding internal instead of presenting it as a shipped surface.
