# Project Decisions

Living document for durable architectural and workflow decisions.

Entry format:
- `YYYY-MM-DD`: decision — rationale — alternatives or non-goals

Entries:
- `2026-03-13`: Keep `IJepa::forward_step_strict` and `VJepa::forward_step_strict` as the semantic reference paths — `Encoder::Input` is opaque, so the generic trainer cannot pre-mask tokens faithfully — do not treat `jepa_train::JepaComponents::forward_step` as the strict reference implementation.
- `2026-03-13`: Keep checkpoint and ONNX logic inside `jepa-compat` — format-specific code stays isolated from core model crates and the CLI remains a thin consumer — avoid spreading loading and runtime concerns across model crates.
- `2026-03-13`: Use typed `thiserror` errors in library crates and reserve `anyhow` for CLI and TUI boundaries — callers need structured failure modes in library code, while user-facing commands need contextual error reporting — do not collapse library errors into opaque CLI-style errors.
- `2026-03-13`: Keep strict parity fixture-driven — Python export tooling is optional maintenance support, not a build dependency — do not make the reference Python environment a requirement for normal Rust verification.
