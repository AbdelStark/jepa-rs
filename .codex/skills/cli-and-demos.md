---
name: cli-and-demos
description: Activate when a task changes clap arguments, command behavior, demo data flow, or the ratatui dashboard in `crates/jepa`. Use this skill for `train`, `encode`, TUI tabs, and reporter-driven demo work so CLI help, tests, and background event handling stay aligned.
prerequisites: cargo
---

# CLI and Demos

<purpose>
Keep the CLI, demos, and TUI aligned with real command behavior, with deterministic smoke paths and explicit reporter updates.
</purpose>

<context>
- `crates/jepa/src/cli.rs` defines clap arguments and parser tests.
- `crates/jepa/src/commands/train.rs` runs real strict image training on CPU with `Autodiff<NdArray<f32>>`, AdamW, and reporter hooks.
- `crates/jepa/src/commands/encode.rs` dispatches between ONNX, safetensors, and deterministic demo inference.
- `crates/jepa/src/tui/app.rs` uses channel-driven background work and tab-specific state machines.
- Demo image data is generated under `target/example-data/`; do not check large datasets into the repo.
</context>

<procedure>
1. Decide whether the change is parser-only, command execution, or TUI state/rendering.
2. Keep `cli.rs` help text, defaults, and parser tests in sync with the implementation.
3. Route long-running demo work through the existing reporter traits and event channels instead of printing directly from worker threads.
4. Keep demo runs small and deterministic. They are smoke paths, not benchmarks.
5. Run `cargo test -p jepa`. If the change affects a shared library behavior, also run that crate's tests.
</procedure>

<patterns>
<do>
- Update clap parser tests whenever a flag, default, or alias changes.
- Use `anyhow::Context` for user-facing command failures in `crates/jepa`.
- Keep training and inference summaries small enough for the TUI to stream and display.
</do>
<dont>
- Do not introduce new datasets or heavy assets into the repo for demo purposes.
- Do not bypass reporter hooks in background tasks; the TUI depends on those events.
- Do not let command help text drift from real defaults.
</dont>
</patterns>

<examples>
Example: parser guard for mutually exclusive train inputs.
```rust
let cli = Cli::try_parse_from([
    "jepa",
    "train",
    "--dataset",
    "train.safetensors",
    "--dataset-dir",
    "./images",
]);
assert!(cli.is_err());
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| A new flag parses incorrectly | `cli.rs` and the command implementation drifted | Update clap definitions and parser tests together |
| TUI run appears stuck in `Running` | A completion or failure event was not sent | Check the worker thread path in `tui/app.rs` and ensure the reporter emits a terminal event |
| safetensors encode fails after a preset change | Model weights no longer match the preset dimensions | Re-check preset selection and checkpoint loading path |
</troubleshooting>

<references>
- `crates/jepa/src/cli.rs`: clap surface and parser tests
- `crates/jepa/src/commands/train.rs`: real training command and reporters
- `crates/jepa/src/commands/encode.rs`: ONNX, safetensors, and demo encode paths
- `crates/jepa/src/tui/app.rs`: TUI state and event handling
</references>
