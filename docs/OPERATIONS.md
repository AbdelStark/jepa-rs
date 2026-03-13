# Operations Runbooks

This document is the operator-facing guide for verification, package smoke, and
known limitations. It is intentionally written so downstream users do not need
to read source before they can diagnose common failures.

## Support Boundary

- Supported semantic reference paths are [`IJepa::forward_step_strict`](../crates/jepa-vision/src/image.rs) and [`VJepa::forward_step_strict`](../crates/jepa-vision/src/video.rs).
- [`JepaComponents::forward_step`](../crates/jepa-train/src/trainer.rs) remains an approximate orchestration helper. It does not hide target tokens before encoder self-attention.
- Differential parity currently covers strict I-JEPA image flows only. It does not prove strict video parity.
- ONNX support covers metadata inspection and initializer loading only. It does not execute ONNX graphs.
- Non-goals for the first release candidate: distributed training, runtime ONNX execution, a model zoo, and production-scale performance tuning.

## Parity Triage

Primary command:

```bash
scripts/run_parity_suite.sh
```

Single-fixture rerun:

```bash
scripts/run_parity_suite.sh specs/differential/ijepa_strict_rect_fixture.json
```

Triage steps:

1. Confirm `python3` is available. The Rust test shells out to the fixture renderer before comparing tensors.
2. Re-run the failing fixture alone so the failure is isolated to one exported case.
3. If the failure is `target position ... exceeds predictor capacity`, set the fixture predictor `max_target_len` to at least `max(mask.target_indices) + 1`.
4. If the failure is a shape mismatch, compare the fixture `config.encoder`, `config.predictor`, and `mask.total_tokens` against the expected image grid and patch size.
5. If only one bundled fixture fails, inspect the coverage unique to that case first:
   `ijepa_strict_rect_fixture.json` stresses non-square grids and deeper stacks.
   `ijepa_strict_rgb_patch24_fixture.json` stresses RGB input and asymmetric patches.
6. If all fixtures fail in the same stage, suspect a shared regression in patch embedding, RoPE, masked gather semantics, transformer math, or predictor target-position handling.
7. Regenerate the checked-in fixtures only after understanding the drift. Use `python3 specs/differential/export_ijepa_strict_fixture.py` and review the metadata and tolerance changes before accepting them.

## Package Smoke Troubleshooting

Dependency-order package smoke:

```bash
cargo package -p jepa-core --no-verify
cargo package -p jepa-vision --no-verify --exclude-lockfile
cargo package -p jepa-world --no-verify --exclude-lockfile
cargo package -p jepa-train --no-verify --exclude-lockfile
cargo package -p jepa-compat --no-verify --exclude-lockfile
cargo package -p jepa --no-verify --exclude-lockfile
```

Common failures:

- Registry-resolution failure for downstream crates:
  Re-run with `--exclude-lockfile` while the workspace is still unpublished. Cargo otherwise tries to resolve internal crate versions from crates.io when it creates the packaged lockfile.
- Dirty worktree during a local rehearsal:
  Use `--allow-dirty` only for the rehearsal command. The tagged release path should run from a clean checkout.
- Missing readme, license, or documentation paths:
  Check the crate manifest `readme`, `license`, `repository`, and `documentation` fields. `cargo package --list -p <crate>` is useful when packaged contents look wrong.
- Internal version mismatch:
  Ensure downstream workspace dependencies keep explicit `version = "0.1.0"` entries in publishable manifests.

## Known-Limitations Debugging

- Strict versus generic training mismatch:
  If strict image or video results disagree with the generic trainer path, treat the strict path as the semantic reference. The generic helper is approximate by design.
- Predictor position errors:
  `TransformerPredictor` expects real flattened token indices. Passing `[0, 1, 2]` for a mask whose real target indices are `[5, 6, 7]` is a bug.
- Parity coverage expectations:
  The bundled suite proves three strict image flows. It does not prove every image shape, every predictor depth, or any video flow.
- ONNX execution expectations:
  `jepa-compat` can inspect model metadata and import initializers. If you need to execute an ONNX graph, that is a separate scope decision and not a supported workflow in this repository today.

## Release Rollback Pointer

Rollback and partial-publish handling are documented in [`docs/RELEASE.md`](./RELEASE.md). Do not attempt to reuse a published version number after a failed release sequence.
