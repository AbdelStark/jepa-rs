---
name: compat-loading
description: Guide for safetensors checkpoint loading, ONNX metadata/initializer inspection, PyTorch key remapping, and pretrained model discovery in jepa-rs. Activate when working with checkpoint files, model loading, weight conversion, or ONNX integration.
prerequisites: safetensors 0.7, half 2 workspace dependencies
---

# Compat & Loading

<purpose>
Load pretrained weights and inspect model formats. The compat crate bridges external formats (safetensors, ONNX) into burn's type system.
Critical boundary: ONNX metadata/initializer loading works. Full ONNX graph execution exists via tract but is not production-proven.
</purpose>

<context>
jepa-compat modules:

| Module | Capability | Status |
|--------|-----------|--------|
| safetensors.rs | Load .safetensors checkpoints, F16/BF16→f32 widening | Functional |
| keymap.rs | Remap PyTorch weight keys to burn naming | Functional |
| onnx.rs | Parse ONNX ModelProto, extract metadata/shapes/dtypes, load initializers | Functional |
| runtime.rs | Execute ONNX graphs via tract, OnnxEncoder adapter | Exists, not production-proven |
| registry.rs | Discover pretrained models | Functional |

Key types:
- `LoadedTensor`: Contains original key, mapped key, tensor data, original dtype
- `OnnxModelInfo`: Parsed ONNX metadata (inputs, outputs, shapes, dtypes)
- `OnnxDtype`: Float32, Float16, BFloat16, Int64, Int32, etc.
- `OnnxSession`: Tract-backed runtime session
- `OnnxEncoder<B>`: Wraps OnnxSession as `Encoder<B>` for trait compatibility
</context>

<procedure>
Loading a safetensors checkpoint:

1. Use `safetensors::SafeTensors::deserialize(&bytes)` to open the file
2. Iterate tensors, checking dtype (F32, F16, BF16)
3. For F16/BF16: widen to f32 using `half` crate before creating burn tensors
4. Apply key remapping via `keymap` module for PyTorch→burn naming
5. Return `LoadedTensor` structs with both original and mapped keys

Inspecting an ONNX model:

1. Use `OnnxModelInfo::from_file(path)` — returns metadata without executing
2. Access `.inputs`, `.outputs` for I/O specs (name, shape, dtype)
3. Access initializers for weight inspection
4. Handle errors: `OnnxError::FileNotFound` vs `OnnxError::ParseError`

ONNX runtime (if approved for use):

1. Create `OnnxSession::from_file(path)`
2. Run inference: `session.run_f32(&shape, &data)` → `InferenceOutput { data, shape }`
3. Or wrap as encoder: `OnnxEncoder::new(session)` implements `Encoder<B>`
</procedure>

<patterns>
<do>
  — Always handle dtype conversion explicitly (F16→f32, BF16→f32)
  — Use key remapping for PyTorch weights — naming conventions differ
  — Distinguish missing file errors from parse/runtime errors in ONNX
  — Test with real .safetensors fixtures when available
  — Document ONNX scope limitations in any code that uses it
</do>
<dont>
  — Don't claim ONNX runtime works for production without explicit approval
  — Don't assume tensor shapes match — validate against model config
  — Don't skip key remapping — PyTorch and burn use different weight naming
  — Don't load large checkpoints in unit tests — use small fixtures
</dont>
</patterns>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| `LoadError::ShapeMismatch` | Checkpoint tensor shape doesn't match model | Check model config dimensions match checkpoint |
| `OnnxError::FileNotFound` vs `OnnxError::ParseError` | Different error paths | from_file distinguishes these — handle separately |
| F16 tensors loaded as garbage | Missing dtype widening | Use half crate to convert F16→f32 before creating burn tensor |
| Key not found after remapping | PyTorch naming convention changed | Check keymap.rs remapping rules, add new mapping if needed |

</troubleshooting>

<references>
— crates/jepa-compat/src/safetensors.rs: Checkpoint loading with dtype widening
— crates/jepa-compat/src/onnx.rs: ONNX metadata inspection and initializer loading
— crates/jepa-compat/src/runtime.rs: Tract-backed ONNX execution
— crates/jepa-compat/src/keymap.rs: PyTorch → burn key remapping rules
— crates/jepa-compat/src/registry.rs: Pretrained model discovery
</references>
