# WebGPU Browser Demo — Feasibility Analysis & Specification

> **Status**: Research / Specification
> **Date**: 2026-03-14
> **Scope**: Full in-browser JEPA demo with training and inference via WebGPU

---

## 1. Executive Summary

Building a web-based demo of jepa-rs that runs **both training and inference
in the browser** using WebGPU is **feasible** with constraints. Burn 0.20.1
already supports WASM + WebGPU compilation via `burn-wgpu`, and the jepa-rs
codebase is backend-generic by design (`B: Backend`). However, browser-based
**training** is uncharted territory for the burn ecosystem (existing demos are
inference-only), and WebGPU itself imposes hard memory and performance limits.

**Verdict**: A compelling demo is achievable using a tiny model preset
(`tiny_test`: ~12K params) with synthetic data, showing live training loss
curves and inference visualization — all running client-side with zero
server dependencies.

---

## 2. Technology Stack Assessment

### 2.1 Burn Framework WebGPU Support

| Aspect | Status | Notes |
|--------|--------|-------|
| `burn-wgpu` WASM target | **Supported** | Compiles to `wasm32-unknown-unknown` |
| WebGPU browser backend | **Supported** | Via wgpu → WebGPU API |
| `burn-ndarray` WASM target | **Supported** | CPU fallback, `no_std` capable |
| `burn-autodiff` on WGPU | **Architecturally supported** | `Autodiff<Wgpu>` composes like any backend |
| `burn-train` Learner | **Not WASM-compatible** | Uses std threading, file I/O, signal handling |
| Custom training loops | **Compatible** | Manual optimizer step loops work in WASM |

**Key insight**: `burn-train`'s `Learner` abstraction is NOT wasm-compatible,
but the lower-level components (modules, optimizers, autodiff, tensors) all
are. jepa-rs already uses a custom training loop in `commands/train.rs`, not
the `Learner`, so this is not a blocker.

### 2.2 WebGPU Browser Support (as of March 2026)

| Browser | WebGPU Status |
|---------|---------------|
| Chrome/Edge 113+ | Stable since April 2023 |
| Firefox | Nightly only, ~90% spec compliance |
| Safari 26+ | Supported since beta (June 2025) |

### 2.3 WebGPU Limitations for ML

| Constraint | Impact | Mitigation |
|-----------|--------|------------|
| GPU memory: ~4-6 GB practical limit | Cannot run large ViT models | Use `tiny_test` (~12K) or `vit_small_patch16` (~22M) |
| No native threading in WASM | Training loop is single-threaded | Acceptable for demo scale |
| Buffer size limits | Large tensors may fail on some GPUs | Keep batch size small (1-4) |
| First-run shader compilation stall | 1-3s pause on first inference | Show loading indicator |
| HTTPS required | Must serve over HTTPS or localhost | Standard deployment practice |
| No filesystem access | Cannot load images from disk | Use synthetic data or canvas/upload |

---

## 3. Architecture

### 3.1 Crate Organization

```text
crates/
└── jepa-web/                    # New crate (wasm-only binary)
    ├── Cargo.toml               # burn + burn-wgpu with wasm features
    ├── src/
    │   ├── lib.rs               # wasm-bindgen entry points
    │   ├── backend.rs           # Backend type aliases + device init
    │   ├── training.rs          # Browser-adapted training loop
    │   ├── inference.rs         # Inference API exposed to JS
    │   └── state.rs             # Shared model/training state
    ├── index.html               # Trunk entry point
    ├── js/
    │   ├── app.js               # UI orchestration
    │   ├── charts.js            # Loss/metric visualization
    │   └── canvas.js            # Image input/output rendering
    └── assets/
        └── style.css
```

### 3.2 Backend Configuration

```rust
// For WebGPU-capable browsers
type WebBackend = Autodiff<Wgpu>;

// CPU fallback for browsers without WebGPU
type CpuBackend = Autodiff<NdArray<f32>>;
```

The demo should detect WebGPU availability at runtime and fall back to
`NdArray` (CPU-via-WASM) if unavailable, with a warning about reduced
performance.

### 3.3 Component Reuse from Existing Crates

| Component | Source Crate | Web Compatibility |
|-----------|-------------|-------------------|
| `VitConfig` presets | `jepa-vision` | Direct reuse — generic over `B` |
| `VitEncoder` | `jepa-vision` | Direct reuse |
| `IJepa` / `IJepaConfig` | `jepa-vision` | Direct reuse |
| `TransformerPredictor` | `jepa-vision` | Direct reuse |
| `BlockMasking` | `jepa-core` | Direct reuse |
| `L2Energy`, `CosineEnergy` | `jepa-core` | Direct reuse |
| `VICReg`, `BarlowTwins` | `jepa-core` | Direct reuse |
| `Ema` | `jepa-core` | Direct reuse |
| `AdamW` optimizer | `burn` | Direct reuse |
| `WarmupCosineSchedule` | `jepa-train` | Direct reuse |
| Image loading/preprocessing | `crates/jepa` | **Needs adaptation** — uses `image` crate with filesystem |
| `Learner` / trainer | `burn-train` | **Not compatible** — needs custom WASM loop |
| Checkpoint save/load | `jepa-train` | **Partial** — serialize to IndexedDB instead of files |
| TUI dashboard | `crates/jepa` | **Not applicable** — replaced by HTML/JS UI |

**~80% of the ML pipeline is directly reusable** without modification because
the codebase is generic over `B: Backend`.

### 3.4 Data Flow

```text
Browser UI                          WASM Module (Rust)
──────────                          ──────────────────

[Start Training] ──────────────►    init_training(config)
                                        │
                                        ▼
                                    Create VitEncoder<Autodiff<Wgpu>>
                                    Create IJepa model
                                    Create AdamW optimizer
                                    Create BlockMasking
                                        │
                                        ▼
[Step Button / Timer] ─────────►    train_step()
                                        │
                                        ├─ Generate synthetic batch
                                        ├─ Generate mask (BlockMasking)
                                        ├─ Forward pass (IJepa::forward_step_strict)
                                        ├─ Backward pass (autodiff)
                                        ├─ Optimizer step (AdamW)
                                        ├─ EMA update
                                        │
                 ◄──────────────────────┘
[Update Charts]                     Return { loss, energy, reg, lr, step }

[Upload/Draw Image] ───────────►    run_inference(image_data)
                                        │
                                        ├─ Preprocess to tensor
                                        ├─ Encode (VitEncoder::forward)
                                        │
                 ◄──────────────────────┘
[Show Embeddings]                   Return { embeddings, patch_norms, stats }
```

---

## 4. Demo Specification

### 4.1 Training Tab

**Model**: `VitConfig::tiny_test()` — 2 layers, dim=32, 2 heads, ~12K params

**Data source**: Synthetic random tensors (no filesystem needed), matching
the existing `SyntheticTraining` demo pattern.

**UI elements**:
- Model preset selector (tiny_test only for training; larger for inference-only)
- Hyperparameter controls: learning rate, batch size, steps, EMA momentum
- Start / Pause / Reset buttons
- Live charts (via Chart.js or similar):
  - Total loss over steps
  - Energy loss vs regularization loss
  - Learning rate schedule curve
  - EMA momentum curve
- Step counter and elapsed time
- Current mask visualization (context vs target patches as a grid)

**Training loop**: Uses `requestAnimationFrame` or `setInterval` to yield
back to the browser between steps, preventing UI freezes.

### 4.2 Inference Tab

**Model**: Either the just-trained model or a fresh random-init model
at any preset size (up to `vit_base_patch16` for WebGPU, `vit_small_patch16`
for CPU fallback).

**Input sources**:
1. Built-in synthetic demo patterns (gradient, checkerboard, rings, etc. —
   reusing the existing `demo_pattern_images()` logic)
2. User-uploaded image (via `<input type="file">` or drag-and-drop)
3. Canvas drawing (freehand, like Burn's MNIST demo)

**Output visualization**:
- Per-patch representation norms as a heatmap overlay
- Embedding statistics (mean, std, min, max)
- Token-level cosine similarity matrix
- Latency per inference call

### 4.3 Architecture Visualization Tab

**Static/interactive diagram** showing:
- JEPA architecture (context encoder → predictor → target encoder)
- Data flow with mask overlay
- EMA relationship between encoders
- Clickable components linking to the relevant Rust source modules

---

## 5. Implementation Plan

### Phase 1: Skeleton (1-2 days)

1. Create `crates/jepa-web/` with `Cargo.toml` targeting `wasm32-unknown-unknown`
2. Configure `burn` + `burn-wgpu` with WASM features
3. Expose a minimal `#[wasm_bindgen]` function: create a `VitEncoder<Wgpu>`,
   run one forward pass on random data, return embedding stats
4. Set up `trunk` build pipeline with `index.html`
5. Verify it runs in Chrome with WebGPU

### Phase 2: Inference Demo (2-3 days)

1. Wire up image upload → preprocessing → encoder forward → display results
2. Add synthetic demo patterns from existing `demo.rs` (adapted for WASM)
3. Build patch-norm heatmap visualization
4. Add model preset selector
5. Implement CPU fallback detection + `NdArray` backend path

### Phase 3: Training Demo (3-5 days)

1. Port the training loop from `commands/train.rs` to an async-yield pattern
2. Wire up `Autodiff<Wgpu>` backend with `AdamW` optimizer
3. Implement step-by-step execution with JS-side scheduling
4. Build live loss chart with streaming data from WASM
5. Add mask visualization (context/target patch grid)
6. Add hyperparameter controls and start/pause/reset

### Phase 4: Polish (2-3 days)

1. Architecture diagram tab
2. Responsive layout and mobile support
3. Performance profiling and optimization
4. Error handling and browser compatibility warnings
5. Deployment pipeline (GitHub Pages or similar)

**Estimated total**: 8-13 days of focused development

---

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `Autodiff<Wgpu>` fails to compile to WASM | Low | Blocking | Fall back to `Autodiff<NdArray>` for training; keep inference on Wgpu |
| Training is too slow in browser | Medium | Demo quality | Use `tiny_test` preset (~12K params); show it's real training, not fast training |
| WASM binary too large | Medium | UX | Use `wasm-opt`, strip debug info; target <5MB |
| `burn-wgpu` WASM compilation hits recursion limits | Low | Fixable | Add `#![recursion_limit = "256"]` |
| `image` crate doesn't compile to WASM | Medium | Moderate | Use raw pixel arrays from JS canvas; bypass `image` crate in web path |
| WebGPU not available in user's browser | Medium | Partial | CPU fallback with clear performance warning |
| `rand` crate thread_rng unavailable in WASM | Low | Fixable | Use `ChaCha8Rng::seed_from_u64()` — already used in existing code |
| Memory pressure on large models | High for >base | Demo scope | Limit training to tiny/small; allow inference up to base |

---

## 7. Dependencies (New)

| Crate | Purpose | WASM Compatible |
|-------|---------|-----------------|
| `wasm-bindgen` | Rust ↔ JS interop | Yes (core purpose) |
| `web-sys` | Browser API access (console, DOM) | Yes |
| `js-sys` | JavaScript type bindings | Yes |
| `console_error_panic_hook` | Better panic messages in browser | Yes |
| `getrandom` with `js` feature | RNG entropy source in WASM | Yes |
| `serde-wasm-bindgen` | Struct serialization to JS | Yes |

**Note**: Adding dependencies to the workspace requires approval per CLAUDE.md
gating rules. This section documents what would be needed.

---

## 8. Build & Development

### Build Command
```bash
# Install trunk
cargo install trunk

# Build and serve
cd crates/jepa-web
trunk serve --open
```

### Cargo.toml Sketch
```toml
[package]
name = "jepa-web"
edition.workspace = true

[lib]
crate-type = ["cdylib"]

[dependencies]
jepa-core = { path = "../jepa-core" }
jepa-vision = { path = "../jepa-vision" }
jepa-train = { path = "../jepa-train" }
burn = { workspace = true }
burn-wgpu = { workspace = true }
burn-ndarray = { workspace = true }
wasm-bindgen = "0.2"
web-sys = { version = "0.3", features = ["console"] }
serde-wasm-bindgen = "0.6"
console_error_panic_hook = "0.1"
getrandom = { version = "0.3", features = ["js"] }
serde = { workspace = true }
serde_json = { workspace = true }
rand = { workspace = true }
rand_chacha = { workspace = true }

[features]
default = ["webgpu"]
webgpu = []
cpu-only = []
```

---

## 9. Prior Art & References

- [Burn MNIST Inference Web Demo](https://github.com/tracel-ai/burn/tree/main/examples/mnist-inference-web) — inference-only, uses wasm-bindgen
- [Burn Image Classification Web](https://github.com/tracel-ai/burn) — inference with WGPU + WebGPU
- [Burn's Cross-Platform GPU Backend Blog Post](https://burn.dev/blog/cross-platform-gpu-backend/) — architecture of burn-wgpu
- [burn-wgpu crate](https://crates.io/crates/burn-wgpu) — v0.20.1, WebGPU backend
- [Trunk WASM Bundler](https://trunkrs.dev/) — build tool for Rust WASM web apps
- [WebGPU Browser AI Guide](https://aicompetence.org/ai-in-browser-with-webgpu/) — current state of WebGPU for ML
- [WebGPU Bugs Holding Back Browser AI](https://medium.com/@marcelo.emmerich/webgpu-bugs-are-holding-back-the-browser-ai-revolution-27d5f8c1dfca) — practical limitations
- [Client-Side AI in 2025](https://medium.com/@sauravgupta2800/client-side-ai-in-2025-what-i-learned-running-ml-models-entirely-in-the-browser-aa12683f457f) — lessons learned

---

## 10. Open Questions

1. **Should training run on `Autodiff<Wgpu>` or `Autodiff<NdArray>`?**
   `Autodiff<Wgpu>` is the aspirational target (GPU-accelerated training in
   browser), but `Autodiff<NdArray>` is safer and proven. A feature flag
   could support both.

2. **Should the web crate live in the workspace or be standalone?**
   Workspace membership is simpler for path dependencies, but adds WASM
   target requirements to CI. A separate workspace with path deps might
   be cleaner.

3. **UI framework**: Pure JS/HTML vs Yew/Leptos/Dioxus (Rust WASM UI frameworks)?
   Pure JS is simpler, faster to develop, and avoids doubling the WASM binary
   size. Recommended for v1.

4. **Pre-trained weights**: Should the demo ship with pre-trained checkpoint
   weights for more meaningful inference, or is random-init sufficient for
   demonstrating the architecture? Pre-trained weights add download size
   but dramatically improve the inference demo's educational value.

---

## 11. Success Criteria

- [ ] User opens a URL, sees JEPA training loss decreasing in real-time in their browser
- [ ] Zero server-side computation — everything runs client-side
- [ ] Training uses the real jepa-rs code paths (masking, energy, EMA, optimizer)
- [ ] Inference produces and visualizes patch-level embeddings
- [ ] Works in Chrome stable with WebGPU; graceful fallback for other browsers
- [ ] WASM bundle < 5 MB (excluding any pre-trained weights)
- [ ] Page load to first training step < 5 seconds on modern hardware
