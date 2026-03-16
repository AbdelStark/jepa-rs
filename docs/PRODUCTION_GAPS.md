# Production Gaps

This register is intentionally blunt. It tracks what still blocks an unqualified
"production ready" claim.

## Verdict Today

`jepa-rs` is not yet production ready as a whole project. The library and CLI
surfaces are in better shape than the browser and runtime-adapter surfaces, but
the repo still has correctness and release-confidence gaps that should be fixed
before claiming that standard publicly.

## Ranked Gaps

| Severity | Gap | Why it matters | Current state |
|----------|-----|----------------|---------------|
| Red | Strict video parity is still missing | V-JEPA can drift semantically without a fixture-based backstop | Image parity exists; video parity is pending |
| Red | ONNX graph execution is not production-grade | Callers can assume runtime inference is as trustworthy as metadata loading when it is not | Metadata and initializer loading are supported; tract execution remains a prototype path |
| Orange | Browser demo exported path is CPU-backed only | The repo has WebGPU scaffolding, but callers should not infer a validated GPU browser runtime | Docs now call this out; runtime selection is still future work |
| Orange | Release smoke does not cover `jepa` and `jepa-web` packaging | Published surfaces can regress outside the current package-smoke safety net | CI package smoke covers only the core library crates |
| Orange | Performance expectations are undocumented | Criterion benchmarks exist, but no budget or regression threshold tells users what "fast enough" means | Bench smoke exists; no enforced SLO-style budget |

## Non-Goals For This Repo

Some classic production-service requirements do not apply directly here because
this project is not a network service:

- No HTTP health endpoint
- No request tracing or distributed trace IDs
- No rollout controller or service rollback orchestration

The relevant standard here is truthful library behavior, strong regression
coverage, repeatable release gates, and explicit limitations.
