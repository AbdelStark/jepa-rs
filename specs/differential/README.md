# Differential Fixtures

The Rust workspace keeps the differential-parity interface fixture-driven so
the reference Python environment does not become an implicit build dependency.

## Expected Flow

1. Export a fixture from a canonical Python JEPA implementation.
2. Store the fixture outside the repository or in a dedicated fixture bucket.
3. Run:

```bash
scripts/run_parity_suite.sh /path/to/ijepa-reference-fixture.json
```

## Fixture Scope

The intended fixture should capture at least one strict image JEPA flow:

- raw input tensor
- mask specification
- target positions
- reference context representation
- reference target representation
- reference predictor output
- reference energy value

The current repository does not bundle a canonical reference fixture. That
keeps the Rust workspace self-contained while still defining a concrete parity
entry point for local and CI provisioning.
