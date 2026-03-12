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

Reference parity is exercised through exported fixture workflows rather than
hard-coding Python dependencies into the Rust workspace.

Current command surface:

```bash
scripts/run_parity_suite.sh /path/to/ijepa-reference-fixture.json
```

The fixture is expected to come from the canonical Python reference stack.
The script is intentionally separate so CI can keep parity optional until the
reference environment is provisioned.
