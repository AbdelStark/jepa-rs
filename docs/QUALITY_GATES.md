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
Maintained benchmark budgets and the capture workflow live in
[`docs/PERFORMANCE.md`](./PERFORMANCE.md).

Smoke command:

```bash
cargo bench --workspace --no-run
```

Baseline workflow:

```bash
cargo bench -p jepa-core --bench core_bench -- --save-baseline main
cargo bench -p jepa-vision --bench vision_bench -- --save-baseline main
cargo bench -p jepa-train --bench train_bench -- --save-baseline main
cargo bench -p jepa-world --bench world_bench -- --save-baseline main
```

Review policy:

- Benchmark-affecting PRs should mention expected hot-path impact.
- Maintained release budgets currently cover masking, strict image forward, trainer orchestration, and world planning hot paths.
- Regressions above the thresholds in [`docs/PERFORMANCE.md`](./PERFORMANCE.md) need an explanation before release.
- Correctness wins over speed, but unexplained slowdowns do not ship silently.

## Differential Parity

Reference parity is exercised through exported fixtures rather than hard-coding
the Python reference environment into the Rust workspace.

Primary command:

```bash
scripts/run_parity_suite.sh
```

Override the bundled fixture set when you want to validate a different export:

```bash
scripts/run_parity_suite.sh /path/to/ijepa-reference-fixture.json
```

Or point the runner at a directory of fixtures:

```bash
scripts/run_parity_suite.sh /path/to/fixture-directory
```

Current policy:

- CI runs `scripts/run_parity_suite.sh` against every checked-in fixture in [`specs/differential/`](../specs/differential/).
- The parity command requires `python3` to decode the exported JSON fixture into the Rust-side comparator.
- Bundled per-fixture tolerances are documented in [`specs/differential/README.md`](../specs/differential/README.md).
- The current checked-in suite covers three strict I-JEPA image flows; strict video parity remains out of scope for this release candidate.

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
- Troubleshooting for package smoke failures lives in [`docs/OPERATIONS.md`](./OPERATIONS.md).
