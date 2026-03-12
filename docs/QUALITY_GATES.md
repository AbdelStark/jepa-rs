# Quality Gates

This repository treats production readiness as an enforced verification
surface, not just a passing unit-test suite.

## Required Local Checks

Run these from the workspace root:

```bash
cargo fmt -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --doc
cargo doc --workspace --no-deps
scripts/run_parity_suite.sh
```

## Coverage Policy

Coverage is enforced in CI with `cargo-llvm-cov`.

Local command:

```bash
cargo llvm-cov --workspace --all-features --fail-under-lines 80
```

Policy:

- New features should add tests in the same change.
- Coverage drops below the threshold should block release work.
- Exceptions should be documented in the PR, not left implicit.

## Fuzzing

Dedicated fuzz targets live under [`fuzz/`](../fuzz).

Smoke commands:

```bash
(cd fuzz && cargo fuzz run masking -- -runs=1000)
(cd fuzz && cargo fuzz run gather -- -runs=1000)
(cd fuzz && cargo fuzz run energy -- -runs=1000)
(cd fuzz && cargo fuzz run checkpoint_parsing -- -runs=1000)
```

Targets:

- `masking`: block, spatiotemporal, and multi-block mask generation
- `gather`: `Representation::gather` on variable shapes and masks
- `energy`: L2, cosine, and Smooth L1 energy paths
- `checkpoint_parsing`: safetensors and ONNX parser surfaces

## Benchmarks

Criterion benchmarks are checked in under the crate `benches/` directories.

Smoke command:

```bash
cargo bench --workspace --no-run
```

Baseline workflow:

```bash
cargo bench -p jepa-core --bench core_bench -- --save-baseline main
cargo bench -p jepa-train --bench train_bench -- --save-baseline main
cargo bench -p jepa-world --bench world_bench -- --save-baseline main
```

Review policy:

- Benchmark-affecting PRs should mention expected hot-path impact.
- Regressions larger than 5% on maintained baselines need an explanation.
- Correctness wins over speed, but unexplained slowdowns do not ship silently.

## Differential Parity

Reference parity is exercised through exported fixtures rather than hard-coding
the Python reference environment into the Rust workspace.

Primary command:

```bash
scripts/run_parity_suite.sh
```

Override the bundled fixture when you want to validate a different export:

```bash
scripts/run_parity_suite.sh /path/to/ijepa-reference-fixture.json
```

Current policy:

- CI runs `scripts/run_parity_suite.sh` against the checked-in strict image fixture at [`specs/differential/ijepa_strict_tiny_fixture.json`](../specs/differential/ijepa_strict_tiny_fixture.json).
- The parity command requires `python3` to decode the exported JSON fixture into the Rust-side comparator.
- The bundled fixture compares strict I-JEPA context, target, predicted, and energy outputs with `abs_tolerance=1e-5` and `rel_tolerance=1e-5`.
- Larger or additional reference fixtures remain optional until maintainers provision them explicitly.

## Package Smoke

Release-candidate hardening includes a packaging dry-run for each crate:

```bash
cargo package -p jepa-core --no-verify
cargo package -p jepa-vision --no-verify --exclude-lockfile
cargo package -p jepa-world --no-verify --exclude-lockfile
cargo package -p jepa-train --no-verify --exclude-lockfile
cargo package -p jepa-compat --no-verify --exclude-lockfile
```

Policy:

- CI runs the package smoke commands above on every change.
- `--exclude-lockfile` is required for downstream unpublished workspace crates so Cargo does not try to resolve internal crate versions from crates.io before the staged publish happens.
- Once the crates are published, maintainers may optionally rerun downstream packaging without `--exclude-lockfile` as an extra registry-resolution check.
