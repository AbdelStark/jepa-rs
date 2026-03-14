# Causal-JEPA (C-JEPA) — Feature Specification

> **Status**: Implemented (all 5 phases)
> **Date**: 2026-03-14
> **Paper**: [Causal-JEPA: Learning World Models through Object-Level Latent Interventions](https://arxiv.org/abs/2602.11389)
> **Authors**: Nam, Le Lidec, Maes, LeCun, Balestriero (2025)
> **Reference impl**: <https://github.com/galilai-group/cjepa>

---

## 1. Paper Summary

C-JEPA extends JEPA to **object-centric representations**. Instead of masking
spatial patches (I-JEPA) or spatiotemporal tubes (V-JEPA), it masks entire
*objects* — forcing the predictor to reason about inter-object interactions.
The authors prove this creates "latent interventions" that induce causal
inductive bias, enabling counterfactual reasoning and efficient planning.

### Key results

| Benchmark | Result |
|-----------|--------|
| CLEVRER counterfactual QA | ~20% improvement over baselines |
| Push-T robotic manipulation | Comparable to patch-based world models |
| Token efficiency vs DINO-WM | **98% reduction** (1.02% of patch tokens) |
| Planning speed vs patch models | **8x faster** |

---

## 2. Architecture

C-JEPA has three components, departing from the standard JEPA two-encoder
pattern:

```text
┌─────────────────────────────────────────────────────────┐
│                       C-JEPA                            │
│                                                         │
│  ┌──────────────────┐                                   │
│  │  Frozen Object   │   Video frames                    │
│  │  Encoder          │──────────► Per-object slots       │
│  │  (VideoSAUR on   │           z_i ∈ R^128             │
│  │   DINOv2 ViT-S)  │           N slots per frame       │
│  └──────────────────┘                                   │
│           │                                             │
│           ▼                                             │
│  ┌──────────────────┐   ┌──────────────────────┐        │
│  │  Object-Level    │   │  Auxiliary Encoders   │        │
│  │  Masking         │   │  (actions, proprio)   │        │
│  │                  │   │  1D convolutions       │        │
│  │  Mask whole      │   └──────────┬───────────┘        │
│  │  object slots    │              │                    │
│  └────────┬─────────┘              │                    │
│           │                        │                    │
│           ▼                        ▼                    │
│  ┌──────────────────────────────────────────────┐       │
│  │  ViT-Style Masked Predictor                  │       │
│  │  6 layers, 16 heads, head_dim=64, MLP=2048   │       │
│  │                                              │       │
│  │  Input: visible slots + action/proprio tokens│       │
│  │  Output: predicted masked object states      │       │
│  └──────────────────────────────────────────────┘       │
│                        │                                │
│                        ▼                                │
│               MSE loss (history + future)               │
└─────────────────────────────────────────────────────────┘
```

### Key differences from I-JEPA / V-JEPA

| Aspect | I-JEPA / V-JEPA | C-JEPA |
|--------|-----------------|--------|
| Masking unit | Spatial patches / spatiotemporal tubes | Whole objects |
| Encoder training | EMA self-supervised | Frozen pretrained backbone |
| Target encoder | EMA copy of context encoder | Same frozen encoder |
| Input granularity | Patch tokens (~196-1568 per frame) | Object slots (~4-7 per frame) |
| Action conditioning | None | Actions + proprioception via auxiliary encoders |
| Training objective | Predict masked patch reps | Predict masked object states (history + future) |
| Causal reasoning | Not explicit | Formally induced via latent interventions |

---

## 3. Relevance to jepa-rs

### 3.1 High alignment

| jepa-rs component | C-JEPA mapping | Fit |
|-------------------|----------------|-----|
| `MaskingStrategy` trait | Object-level masking variant | **Direct** — `MaskSpec` already supports arbitrary index partitions |
| `ActionConditionedPredictor<B>` | Action-conditioned prediction | **Direct** — trait exists in `jepa-world` |
| `RandomShootingPlanner` (CEM) | CEM-based MPC planning | **Direct** — C-JEPA uses same algorithm |
| `CostFunction<B>` / `L2Cost` | Goal-conditioned planning cost | **Direct** |
| `WorldModel<B, D, C>` | Latent world model for rollouts | **Direct** |
| `Predictor<B>` trait | Masked predictor | **Direct** — predict target from context + positions |

### 3.2 Moderate alignment

| jepa-rs component | C-JEPA mapping | Gap |
|-------------------|----------------|-----|
| `HierarchicalJepa` | C-JEPA is single-level | Could be one level in a hierarchy |
| `Encoder<B>` trait | Frozen slot-attention encoder | Need `SlotAttentionEncoder` impl |
| `Ema` | Not used (frozen encoder) | Training loop needs a `FrozenEncoder` path |
| `VitEncoder` | DINOv2 backbone (frozen, upstream) | Not a standard ViT training target |

### 3.3 Low alignment / new primitives needed

| Primitive | Notes |
|-----------|-------|
| Slot Attention module | Core new component; not in codebase or burn ecosystem |
| Object-identity anchoring | Masked tokens carry identity from first timestep: `z̃ = φ(z_t0) + e_t` |
| Joint history + future loss | `L = L_history + L_future`; differs from single-target JEPA loss |
| Frozen-encoder training loop | No EMA, no gradient through encoder; different from `JepaComponents::forward_step` |
| Auxiliary 1D-conv encoders | Small action/proprioception encoders |

---

## 4. Implementation Plan

### Phase 1: Object-Level Masking (Low effort, high value)

**Crate**: `jepa-core`
**Gated**: Yes — touches `masking.rs` public API

Add an `ObjectMasking` variant to the `MaskingStrategy` trait ecosystem:

```rust
/// Masks entire object slots rather than spatial patches.
///
/// Given N object slots per frame, randomly partitions them into
/// context (visible) and target (masked) subsets.
pub struct ObjectMasking {
    /// Total number of object slots per frame.
    pub num_slots: usize,
    /// Range of objects to mask per frame [min, max].
    pub mask_range: (usize, usize),
}
```

This is straightforward because `MaskSpec` already works with arbitrary
context/target index vectors — no spatial grid assumption is baked in.

**Tests**: Property tests ensuring disjointness, coverage, and range bounds.

### Phase 2: Action-Conditioned Prediction Wiring (Low effort)

**Crate**: `jepa-world`

The `ActionConditionedPredictor` trait already exists. Wire it to accept
object-level representations and produce next-state predictions. Add a
concrete `ObjectDynamicsPredictor<B>` that wraps a transformer predictor
with action/proprioception token injection.

**Tests**: Forward-pass shape tests with mock object representations.

### Phase 3: Slot Attention Encoder (Medium-high effort)

**Crate**: `jepa-vision`

Implement `SlotAttention<B>` as a burn module:

```text
SlotAttention
├── slot_init: LearnableSlotInitializer (N slots × D)
├── k_proj, v_proj, q_proj: Linear layers
├── gru: GRUCell (iterative slot refinement)
├── mlp: FFN (post-attention slot update)
└── iterations: usize (typically 2-3)
```

Then compose with a frozen `VitEncoder` backbone:

```text
VideoSAUREncoder<B>
├── backbone: VitEncoder<B>  (frozen DINOv2 weights)
├── slot_attention: SlotAttention<B>
└── forward(): [B, C, H, W] → [B, N_slots, slot_dim]
```

This is the largest new primitive. Slot attention is well-documented
(Locatello et al. 2020) but has not been implemented in burn before.

**Tests**: Shape tests, slot-count verification, convergence on toy scenes.

### Phase 4: C-JEPA Training Loop (Medium effort)

**Crate**: `jepa-train` (or `jepa-world`)

A `CausalJepaTrainer` that differs from the standard JEPA loop:

1. Frozen encoder — no EMA updates, no gradient through backbone
2. Object-level masking with identity anchoring
3. Joint history + future MSE loss
4. Action/proprioception token injection

Could be implemented as a config variant of the existing training
infrastructure rather than a fully separate system.

**Tests**: Loss-decreasing smoke test on synthetic object trajectories.

### Phase 5: CEM Planning Integration (Low effort)

**Crate**: `jepa-world`

`RandomShootingPlanner` already implements CEM. Connect it to the
C-JEPA dynamics model for planning in object-representation space.
The ~98% token reduction makes this dramatically cheaper than
patch-based planning.

**Tests**: Planning on toy environment with known optimal trajectory.

---

## 5. Effort Estimates

| Phase | Scope | Effort | Priority |
|-------|-------|--------|----------|
| 1. Object-level masking | `jepa-core` | Small | **High** — unlocks the paradigm |
| 2. Action-conditioned wiring | `jepa-world` | Small | **High** — validates existing abstractions |
| 3. Slot attention encoder | `jepa-vision` | Medium-Large | **Medium** — biggest new primitive |
| 4. C-JEPA training loop | `jepa-train`/`jepa-world` | Medium | **Medium** — after slot attention |
| 5. CEM planning integration | `jepa-world` | Small | **Low** — mostly wiring |

---

## 6. Dependencies & Gating

### New modules (autonomous zone)

- `jepa-vision/src/slot_attention.rs` — new file
- `jepa-world/src/object_dynamics.rs` — new file

### Gated changes (require approval)

- `jepa-core/src/masking.rs` — adding `ObjectMasking`
- Any `Cargo.toml` changes if slot attention needs new dependencies
- `jepa-core/src/types.rs` — if `Representation<B>` needs object-aware metadata

### No new external dependencies expected

Slot attention uses only linear layers, softmax, and GRU — all available in
burn 0.20.1. No new crate dependencies anticipated.

---

## 7. Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Slot attention convergence issues in burn | Medium | Start with reference PyTorch impl parity tests |
| GRU not available in burn | Low | Implement as manual gate equations using existing primitives |
| Object-identity anchoring adds complexity to `MaskSpec` | Low | Keep identity anchoring in the masking module, not in `MaskSpec` |
| Frozen-encoder paradigm conflicts with existing EMA loop | Low | Separate training config variant; don't modify existing loop |
| Video-specific data loading for testing | Medium | Use synthetic multi-object trajectories (colored circles) |

---

## 8. Open Questions

1. **Should `ObjectMasking` live in `jepa-core` or `jepa-world`?**
   Core masking strategies are in `jepa-core`, but object-level masking is
   semantically tied to world models. Recommendation: `jepa-core` for
   consistency with `BlockMasking` and `SpatiotemporalMasking`.

2. **Should the frozen-encoder pattern be a first-class training mode?**
   C-JEPA's frozen encoder is a meaningful departure. It could be a
   `TrainConfig::encoder_mode: EncoderMode::Frozen | Ema { momentum }` flag.

3. **How tightly to couple with DINOv2?**
   C-JEPA uses DINOv2 as backbone. jepa-rs could support any frozen
   `Encoder<B>` (including ONNX-loaded models via `jepa-compat`), keeping
   the architecture modular.

4. **Should slot attention be a standalone crate?**
   It's generally useful beyond JEPA. However, adding workspace members
   requires approval. Starting as a module in `jepa-vision` is simpler.

---

## 9. References

- Nam et al. (2025). *Causal-JEPA: Learning World Models through Object-Level
  Latent Interventions*. arXiv:2602.11389.
- Locatello et al. (2020). *Object-Centric Learning with Slot Attention*.
  NeurIPS 2020.
- Zadaianchuk et al. (2023). *VideoSAUR: Self-supervised Video Object
  Discovery with Slot Attention*. NeurIPS 2023.
- Assran et al. (2023). *Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture*. CVPR 2023.
- Bardes et al. (2024). *V-JEPA: Latent Video Prediction for Visual
  Representation Learning*.
- LeCun (2022). *A Path Towards Autonomous Machine Intelligence*.
  OpenReview.
