# Quality Gates

This document is the source of truth for local verification and release-candidate
checks.

## Fast Local Gate

Use the narrowest command that proves the change:

- `cargo test -p jepa-core`
- `cargo test -p jepa-vision`
- `cargo test -p jepa-world`
- `cargo test -p jepa-train`
- `cargo test -p jepa-compat`
- `cargo test -p jepa`
- `cargo test -p jepa-web`

## Shared-Behavior Gate

Run this set when a change crosses crate boundaries, changes public behavior, or
touches docs that describe current behavior:

```bash
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt -- --check
```

## Surface-Specific Additions

Run the extra proof that matches the change:

- Strict image semantics: `scripts/run_parity_suite.sh`
- Browser demo: `cargo test -p jepa-web`
- ONNX, safetensors, or key mapping: `cargo test -p jepa-compat`
- CLI flags or command behavior: `cargo test -p jepa`

## Release-Candidate Gate

These are the checks CI is expected to cover before a release candidate is cut:

```bash
cargo check --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt -- --check
cargo doc --no-deps --all-features
cargo llvm-cov --workspace --all-features --fail-under-lines 80 --summary-only --ignore-filename-regex '(tui/(ui|app|event|mod)\.rs|main\.rs)$'
cargo bench --workspace --no-run
scripts/run_parity_suite.sh
cargo package -p jepa-core --no-verify
cargo package -p jepa-vision --no-verify --exclude-lockfile
cargo package -p jepa-world --no-verify --exclude-lockfile
cargo package -p jepa-train --no-verify --exclude-lockfile
cargo package -p jepa-compat --no-verify --exclude-lockfile
```

If `cargo-audit` is installed locally, also run:

```bash
cargo audit
```

If nightly plus `cargo-fuzz` are installed, mirror the current fuzz smoke jobs:

```bash
(cd fuzz && cargo fuzz run masking --target x86_64-unknown-linux-gnu -- -runs=256)
(cd fuzz && cargo fuzz run gather --target x86_64-unknown-linux-gnu -- -runs=256)
(cd fuzz && cargo fuzz run energy --target x86_64-unknown-linux-gnu -- -runs=256)
(cd fuzz && cargo fuzz run checkpoint_parsing --target x86_64-unknown-linux-gnu -- -runs=256)
```

## Known Gaps In The Gate

- `cargo package` smoke is currently defined only for `jepa-core`,
  `jepa-vision`, `jepa-world`, `jepa-train`, and `jepa-compat`.
- `jepa` and `jepa-web` still require manual release judgment until package
  smoke and publish expectations are documented more deeply.
- Video parity is not yet covered by fixture-driven CI.

## Execution Notes

- Serialize heavy cargo commands when multiple agents share the checkout; Cargo
  file locks are expected otherwise.
- Do not weaken parity fixtures or tolerances to make a gate pass.
- Treat a broken doc link as a failed quality gate when the README,
  CONTRIBUTING guide, or agent context references that path.
