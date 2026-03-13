---
name: checkpoint-and-onnx
description: Activate when a task touches safetensors, key remapping, ONNX metadata, ONNX runtime execution, exported encoder models, or `jepa encode`. Use this skill for interop work because the error modes and verification steps differ from the rest of the workspace.
prerequisites: cargo, python3 for export tooling, optional PyTorch stack for ONNX export
---

# Checkpoint and ONNX

<purpose>
Keep checkpoint loading, key mapping, and ONNX runtime behavior isolated inside `jepa-compat`, with honest error handling and shape validation.
</purpose>

<context>
- `crates/jepa-compat/src/safetensors.rs` handles checkpoint loading and dtype widening.
- `crates/jepa-compat/src/keymap.rs` maps reference Python weight names onto burn-native names.
- `crates/jepa-compat/src/onnx.rs` inspects ONNX metadata and initializers.
- `crates/jepa-compat/src/runtime.rs` runs ONNX graphs through tract and exposes `OnnxEncoder`.
- `crates/jepa/src/commands/encode.rs` is the user-facing entry point for `.onnx` and `.safetensors` models.
- `scripts/export_ijepa_onnx.py` is external tooling. It is not required for normal cargo builds.
</context>

<procedure>
1. Decide which path you are changing: raw safetensors loading, key remapping, ONNX metadata, ONNX runtime, or CLI encode behavior.
2. Keep format-specific logic inside `jepa-compat`. Only touch `crates/jepa` when the user-facing command or output changes.
3. Distinguish error classes clearly: missing file, invalid format, input shape mismatch, and runtime failure should not collapse into one message.
4. For ONNX models with dynamic dims, prefer `OnnxEncoder::from_path_with_input_shape(...)`.
5. Run `cargo test -p jepa-compat`. If CLI encode output or dispatch changed, also run `cargo test -p jepa`.
6. If export assumptions changed, inspect `scripts/export_ijepa_onnx.py` and document external Python requirements instead of assuming they exist.
</procedure>

<patterns>
<do>
- Preserve original checkpoint keys when surfacing unmapped entries or debug data.
- Use the matching `VitConfig` preset when loading burn-native weights into a model.
- Keep ONNX output names and input names stable when they are part of the Rust runtime assumptions.
</do>
<dont>
- Do not claim arbitrary dynamic-shape ONNX execution works unless you tested the exact model and input binding path.
- Do not conflate a missing file with an invalid model.
- Do not add or update ONNX dependencies without approval.
</dont>
</patterns>

<examples>
Example: load an ONNX encoder when the model requires an explicit input shape.
```rust
use jepa_compat::runtime::OnnxEncoder;

let encoder = OnnxEncoder::from_path_with_input_shape("model.onnx", &[1, 3, 224, 224])?;
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| `input shape mismatch: expected ... got ...` | The caller shape does not match the optimized ONNX plan | Bind the expected shape explicitly or change the caller input |
| `failed to inject weights from ...` | Preset dimensions and checkpoint tensors do not match | Choose the correct preset or inspect unmapped keys |
| Invalid ONNX bytes parse as load failure | The file is not a valid ONNX model | Reproduce with `cargo test -p jepa-compat` and keep the typed error distinct |
</troubleshooting>

<references>
- `crates/jepa-compat/src/safetensors.rs`: checkpoint loading and validation
- `crates/jepa-compat/src/keymap.rs`: key remapping rules
- `crates/jepa-compat/src/onnx.rs`: metadata and initializer inspection
- `crates/jepa-compat/src/runtime.rs`: tract runtime and `OnnxEncoder`
- `crates/jepa/src/commands/encode.rs`: CLI encode entry point
- `scripts/export_ijepa_onnx.py`: export assumptions and external dependencies
</references>
