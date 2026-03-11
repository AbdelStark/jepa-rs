# jepa-rs

**JEPA primitives in Rust. The first Rust implementation of the Joint Embedding Predictive Architecture.**

[Specification](./SPECIFICATION.md) | [BDD Features](./specs/gherkin/features.feature)

---

## What is this?

`jepa-rs` provides production-grade Rust building blocks for the [Joint Embedding Predictive Architecture (JEPA)](https://openreview.net/pdf?id=BZ5a1r-kVsf), proposed by Yann LeCun in 2022 as the foundation for world models that understand the physical world.

JEPA predicts in representation space rather than pixel space. It's the architecture behind [I-JEPA](https://github.com/facebookresearch/ijepa), [V-JEPA](https://github.com/facebookresearch/jepa), and [V-JEPA 2](https://ai.meta.com/vjepa/), and is the core technology of [AMI Labs](https://amilabs.xyz/).

All existing JEPA implementations are Python/PyTorch. This library brings JEPA to the Rust ecosystem, enabling deployment in safety-critical, embedded, and resource-constrained environments where Python is not an option.

## Why Rust?

- **Safety-critical deployment.** World models will run in healthcare, robotics, and industrial settings. Rust's memory safety guarantees eliminate entire bug classes at compile time.
- **Deterministic execution.** Prerequisite for verifiable and auditable AI. No garbage collector, no runtime surprises.
- **Bare-metal inference.** Run on ARM, RISC-V, wearables, and edge devices. No Python runtime needed.
- **Production infrastructure.** Video preprocessing, data pipelines, model serving at scale.

## Architecture

```
jepa-core     Core traits: Encoder, Predictor, EnergyFn, MaskingStrategy
jepa-vision   Vision Transformer (ViT), patchification, RoPE, I-JEPA, V-JEPA
jepa-world    Action conditioning, planning, H-JEPA, memory
jepa-train    Training loop, schedulers, checkpointing
jepa-compat   Load PyTorch/safetensors weights, ONNX import
```

Backend-agnostic via `burn` framework. Supports CPU (`ndarray`), GPU (`wgpu`, `cuda`), and WASM.

## Status

**Pre-alpha.** Specification phase. See [SPECIFICATION.md](./SPECIFICATION.md) for the complete RFC archive, BDD specs, test strategy, and differential testing plan.

## Reference Implementations

All differential tests run against these Python codebases:

| Repo | Description |
|------|-------------|
| [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) | I-JEPA (images) |
| [facebookresearch/jepa](https://github.com/facebookresearch/jepa) | V-JEPA / V-JEPA 2 (video) |
| [facebookresearch/eb_jepa](https://github.com/facebookresearch/eb_jepa) | EB-JEPA (educational library, includes world model planning) |
| [facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms) | JEPA World Models for physical planning |

## License

Apache-2.0 / MIT dual license.

## Author

[Abdel Bakhta](https://github.com/AbdelStark) ([@AbdelStark](https://x.com/AbdelStark))
