# Skill Registry

Last updated: 2026-03-12

| Skill                 | File                     | Triggers                                          | Priority |
|-----------------------|--------------------------|---------------------------------------------------|----------|
| Implementing RFCs     | implementing-rfcs.md     | implement, RFC, trait, module, specification       | Core     |
| Testing               | testing.md               | test, spec, coverage, proptest, differential, fuzz | Core     |
| Burn Backend          | burn-backend.md          | tensor, backend, burn, ndarray, wgpu, model        | Core     |
| Debugging             | debugging.md             | error, bug, fix, compile, fail, panic              | Core     |

## Missing Skills (Recommended)
- [ ] Performance & benchmarking — criterion setup, profiling numerical code (core_bench.rs exists but is unpopulated)
- [ ] PyTorch compatibility — safetensors loading, weight key mapping (for jepa-compat crate)
- [ ] CI/CD setup — GitHub Actions for Rust workspace (no workflows exist yet)
- [ ] Vision architecture — ViT implementation patterns, patch embedding, RoPE (for jepa-vision)
- [ ] Documentation — rustdoc conventions, doc test patterns (6 doc tests exist as reference)
