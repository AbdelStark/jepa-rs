# Performance Budgets

This repository keeps benchmark smoke in CI, but release readiness requires a
maintained budget and review workflow for the hot paths that matter most to the
current scope.

## Budgeted Surfaces

| Area | Benchmark crate / group | Why it matters | Regression threshold |
| --- | --- | --- | --- |
| Mask generation | `jepa-core` / `masking/block` | JEPA masking runs every training step and is easy to slow down accidentally with extra validation or allocation | `5%` |
| Strict image flow | `jepa-vision` / `vision/ijepa_strict_forward` | This is the strict semantic reference path for image JEPA | `7%` |
| Trainer orchestration | `jepa-train` / `trainer/forward_step` | The generic orchestration path still drives examples and integration work | `5%` |
| Planning | `jepa-world` / `world/cem_planner` and `world/rollout` | Planning cost scales quickly with candidate count and rollout depth | `7%` |

Correctness wins over speed. A change may exceed the threshold if the
slowdown is understood, justified, and called out explicitly in release notes.

## Baseline Capture

Run baseline capture from a clean checkout on a quiet machine with the release
toolchain you plan to ship:

```bash
cargo bench -p jepa-core --bench core_bench -- --save-baseline rc-2026-03-12
cargo bench -p jepa-vision --bench vision_bench -- --save-baseline rc-2026-03-12
cargo bench -p jepa-train --bench train_bench -- --save-baseline rc-2026-03-12
cargo bench -p jepa-world --bench world_bench -- --save-baseline rc-2026-03-12
```

Capture notes alongside the baseline name:

- date and git commit
- Rust toolchain version
- machine / CPU class
- whether the run was done on battery or plugged in

If a benchmark group changes shape materially, capture a new baseline instead of
comparing unrelated measurements.

## Regression Review

Compare a change against the maintained baseline with Criterion's saved-baseline
support:

```bash
cargo bench -p jepa-core --bench core_bench -- --baseline rc-2026-03-12
cargo bench -p jepa-vision --bench vision_bench -- --baseline rc-2026-03-12
cargo bench -p jepa-train --bench train_bench -- --baseline rc-2026-03-12
cargo bench -p jepa-world --bench world_bench -- --baseline rc-2026-03-12
```

Review rules:

- A regression inside the threshold should still be mentioned if it touches a user-visible hot path.
- A regression above the threshold blocks release until the cause is understood.
- If the slowdown buys semantic correctness or a safer API, document that tradeoff instead of hiding it.
- Re-capture the baseline only after the new performance level is accepted deliberately.

## CI And Release Surfacing

- CI continues to run `cargo bench --workspace --no-run` as a compile smoke check.
- Release candidates must run the saved-baseline comparison commands above and summarize the deltas in the release note draft under [`docs/releases/`](./releases).
- Significant regressions are release blockers even if CI bench smoke still compiles cleanly.

## Current Focus

The current benchmark budget is intentionally narrower than the full workspace:

- masking generation in `jepa-core`
- strict I-JEPA image forward in `jepa-vision`
- generic orchestration in `jepa-train`
- rollout and CEM planning in `jepa-world`

Broader optimization work stays secondary to semantic correctness, parity, and
publishability for the first crates.io release.
