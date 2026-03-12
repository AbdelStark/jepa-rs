# Differential Fixtures

The Rust workspace keeps the differential-parity interface fixture-driven so
the reference Python environment does not become an implicit build dependency.

The repository now bundles one strict image-path fixture at
[`ijepa_strict_tiny_fixture.json`](./ijepa_strict_tiny_fixture.json) plus the
Python exporter used to generate it:
[`export_ijepa_strict_fixture.py`](./export_ijepa_strict_fixture.py).

## Expected Flow

1. Use the bundled fixture directly:

```bash
scripts/run_parity_suite.sh
```

2. Or export / provide a different fixture and run:

```bash
scripts/run_parity_suite.sh /path/to/ijepa-reference-fixture.json
```

## Bundled Fixture Scope

The bundled fixture captures one strict image JEPA flow:

- raw input tensor
- mask specification
- target positions
- reference context representation
- reference target representation
- reference predictor output
- reference energy value

It is exported by a small Python reference adapter that mirrors the flattened
token indexing and masked gather semantics used by `facebookresearch/ijepa`
for image-path parity work.

Current tolerances:

- absolute tolerance: `1e-5`
- relative tolerance: `1e-5`
