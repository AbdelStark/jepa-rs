# Differential Fixtures

The Rust workspace keeps the differential-parity interface fixture-driven so
the reference Python environment does not become an implicit build dependency.

The repository bundles three strict image-path fixtures plus the Python exporter
used to generate them:
[`export_ijepa_strict_fixture.py`](./export_ijepa_strict_fixture.py).

## Expected Flow

1. Run the bundled suite directly:

```bash
scripts/run_parity_suite.sh
```

2. Or run a single fixture:

```bash
scripts/run_parity_suite.sh /path/to/ijepa-reference-fixture.json
```

3. Or point the runner at a different fixture directory:

```bash
scripts/run_parity_suite.sh /path/to/fixture-directory
```

## Bundled Fixture Matrix

Each checked-in fixture captures the same strict image JEPA stages:

- raw input tensor
- mask specification
- target positions
- reference context representation
- reference target representation
- reference predictor output
- reference energy value

The exporter mirrors the flattened token indexing and masked gather semantics
used by `facebookresearch/ijepa` for image-path parity work.

| Fixture | Coverage | Absolute tolerance | Relative tolerance |
| --- | --- | --- | --- |
| [`ijepa_strict_tiny_fixture.json`](./ijepa_strict_tiny_fixture.json) | 1-channel image, `2x2` patch grid, 1 encoder layer, 1 predictor layer | `1e-5` | `1e-5` |
| [`ijepa_strict_rect_fixture.json`](./ijepa_strict_rect_fixture.json) | non-square `3x4` patch grid, 2 encoder layers, 2 predictor layers | `2e-5` | `2e-5` |
| [`ijepa_strict_rgb_patch24_fixture.json`](./ijepa_strict_rgb_patch24_fixture.json) | RGB input, asymmetric `2x4` patches, predictor projection `8 -> 12 -> 8` | `2e-5` | `2e-5` |

## Fixture Maintenance

- `scripts/run_parity_suite.sh` discovers every `*_fixture.json` file in this directory by default.
- Regenerate the checked-in fixtures with `python3 specs/differential/export_ijepa_strict_fixture.py`.
- Keep `metadata.name`, `metadata.reference_note`, and the tolerance fields up to date whenever a fixture is replaced.
- Increase tolerances only when a reproduced numeric difference is understood and documented; do not treat tolerance growth as a generic fix for parity failures.
