---
name: strict-vision-models
description: Activate when a task touches `jepa-vision`, masked token semantics, patch or tubelet indexing, target positions, or burn tensor shape logic. Use this for any image or video JEPA change where strict reference behavior versus generic training semantics matters.
prerequisites: cargo, python3
---

# Strict Vision Models

<purpose>
Protect the strict image and video reference paths, preserve mask and token-index semantics, and keep burn tensor shapes correct.
</purpose>

<context>
- `IJepa::forward_step_strict` in `crates/jepa-vision/src/image.rs` and `VJepa::forward_step_strict` in `crates/jepa-vision/src/video.rs` are the semantic reference paths.
- `jepa_train::JepaComponents::forward_step` is intentionally approximate because it cannot pre-mask opaque `Encoder::Input`.
- `TransformerPredictor` consumes real flattened `target_positions`.
- `Representation::gather` preserves masks and should remain the reference gather primitive.
- Strict image parity lives in `crates/jepa-vision/tests/integration.rs` and `specs/differential/*_fixture.json`.
</context>

<procedure>
1. Decide whether the task belongs to the strict path or the approximate generic trainer. If the requirement says `reference`, `faithful`, or `parity`, use the strict path.
2. Start from the owning module: `image.rs` for I-JEPA, `video.rs` for V-JEPA, `vit.rs` for shared encoder behavior, `token_ops.rs` for shared token utilities.
3. Preserve the sequence `mask -> context or target indices -> target_positions -> gather -> predictor`. If any step changes, add a regression test before refactoring further.
4. Check tensor rank after every `reshape`, `slice`, `sum`, or `squeeze`. Most burn shape bugs come from silent rank changes.
5. Run `cargo test -p jepa-vision`. If strict image behavior changed, also run `scripts/run_parity_suite.sh`.
6. If the fix requires changing `jepa-core` wrappers or traits, stop and get approval.
</procedure>

<patterns>
<do>
- Derive patch-grid dimensions from config or `InputShape`; do not duplicate magic numbers.
- Keep public APIs generic over `B: Backend`.
- Use regression tests that prove hidden patches or tubelets cannot leak into strict context encoding.
</do>
<dont>
- Do not treat `JepaComponents::forward_step` as the semantic reference implementation.
- Do not rebuild masks by hand after `Representation::gather`; keep the wrapper semantics intact.
- Do not widen parity tolerances without understanding the numerical difference.
</dont>
</patterns>

<examples>
Example: verify a strict image-path change.
```bash
cargo test -p jepa-vision
scripts/run_parity_suite.sh
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| Predictor rejects target positions | Positions are not real flattened token indices | Pass `mask.target_indices` through unchanged or derive from the same indexing scheme |
| Hidden patches affect strict context output | Full-token encoder behavior leaked into the strict flow | Re-check the masked encode and gather path in `image.rs` or `video.rs` |
| `expected Tensor<_, 3> found Tensor<_, 2>` | A reshape, squeeze, or reduction changed rank | Inspect intermediate dims and restore the expected `[B, S, D]` shape |
</troubleshooting>

<references>
- `crates/jepa-vision/src/image.rs`: strict I-JEPA path and predictor
- `crates/jepa-vision/src/video.rs`: strict V-JEPA path
- `crates/jepa-vision/tests/integration.rs`: strict-path and end-to-end integration tests
- `specs/differential/README.md`: parity fixture workflow
</references>
