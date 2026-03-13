# Lessons Learned

Living document for recurring failure patterns, verified fixes, and reusable debugging shortcuts.

Entry format:
- `YYYY-MM-DD`: lesson — implication or rule

Entries:
- `2026-03-13`: If a change touches mask flow or target indexing, verify both targeted tests and parity fixtures — run crate-local tests and `scripts/run_parity_suite.sh` before considering the change complete.
- `2026-03-13`: New public types often need `src/lib.rs` re-exports — unresolved imports after an otherwise-correct implementation usually mean the public surface was not updated.
- `2026-03-13`: Concurrent cargo commands can block each other in the same checkout — serialize heavy verification jobs when multiple agents share the repo.
