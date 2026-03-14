# Lessons Learned

Living document for recurring failure patterns, verified fixes, and reusable debugging shortcuts.

Entry format:
- `YYYY-MM-DD`: lesson — implication or rule

Entries:
- `2026-03-13`: If a change touches mask flow or target indexing, verify both targeted tests and parity fixtures — run crate-local tests and `scripts/run_parity_suite.sh` before considering the change complete.
- `2026-03-13`: New public types often need `src/lib.rs` re-exports — unresolved imports after an otherwise-correct implementation usually mean the public surface was not updated.
- `2026-03-13`: Concurrent cargo commands can block each other in the same checkout — serialize heavy verification jobs when multiple agents share the repo.
- `2026-03-14`: burn 0.20.1 does not expose `nn::GruCell` — implement GRU gate equations manually with Linear layers and sigmoid/tanh activations when needed (e.g. slot attention refinement).
- `2026-03-14`: `Tensor::squeeze::<N>(dim)` in burn 0.20 takes zero arguments — use `.reshape([...])` with explicit dimensions instead of squeeze for rank reduction.
- `2026-03-14`: When adding new enum variants to `ValueEnum`-derived CLI enums, search for all match statements on that enum across the crate — missing arms cause compile errors in command handlers, not just in the enum definition file.
