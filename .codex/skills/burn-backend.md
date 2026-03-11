---
name: burn-backend
description: Working with the burn 0.16 ML framework in jepa-rs. Activate when writing tensor operations, implementing neural network modules, dealing with backends, or debugging burn-related compilation errors. Also relevant when choosing between backends (ndarray, wgpu, cuda).
prerequisites: burn 0.16 workspace dependency
---

# Burn Backend

<purpose>
Guide for using burn 0.16 correctly in this project.
burn is the ML framework — all tensor math, autograd, and model definitions go through it.
</purpose>

<context>
Key burn concepts used in jepa-rs:

- `Backend` trait: Abstracts compute backend. All types are generic over `B: Backend`.
- `Tensor<B, D>`: D-dimensional tensor. D is const generic (compile-time rank).
- `AutodiffBackend`: Extension of Backend with gradient tracking. Required for training.
- Backends available: `burn_ndarray::NdArray` (CPU), `burn_wgpu::Wgpu` (GPU).

Workspace dependencies:
```toml
burn = { version = "0.16", features = ["autodiff"] }
burn-ndarray = "0.16"   # CPU backend (tests + inference)
burn-wgpu = "0.16"      # GPU backend (training + inference)
```
</context>

<procedure>
Creating tensor-bearing types:

1. Always parameterize on `B: Backend`:
   ```rust
   pub struct MyType<B: Backend> {
       pub data: Tensor<B, 3>,  // 3D tensor
   }
   ```

2. Implement methods with the same generic:
   ```rust
   impl<B: Backend> MyType<B> {
       pub fn new(data: Tensor<B, 3>) -> Self { Self { data } }
   }
   ```

3. For training-specific code, use `AutodiffBackend`:
   ```rust
   pub fn train_step<B: AutodiffBackend>(model: &Model<B>, ...) -> Tensor<B, 1> { ... }
   ```

Creating tensors in tests:

1. Define test backend: `type TestBackend = burn_ndarray::NdArray<f32>;`
2. Get device: `let device = burn_ndarray::NdArrayDevice::Cpu;`
3. Create tensor: `let t = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);`
4. Check shape: `assert_eq!(t.dims(), [2, 2]);`
5. Extract scalar: `let val: f32 = t.into_scalar();`
</procedure>

<patterns>
<do>
  — Use `Tensor<B, D>` with explicit rank D (compile-time shape checking)
  — Use `.dims()` to inspect runtime shape — returns `[usize; D]`
  — Use `Tensor::from_floats(data, &device)` for test data
  — Use `.clone()` when reusing tensors (burn tensors are reference-counted)
  — Use `burn::tensor::activation::*` for activation functions
</do>
<dont>
  — Don't mix backend types — all tensors in an operation must share the same `B`
  — Don't use `.into_data()` in hot paths — it forces a sync and copy
  — Don't assume tensor memory layout — use burn's API, not raw indexing
  — Don't use `f64` tensors unless precision requires it — `f32` is default
</dont>
</patterns>

<troubleshooting>

| Symptom | Cause | Fix |
|---------|-------|-----|
| `the trait Backend is not implemented for...` | Wrong backend type or missing generic | Ensure function is generic over `B: Backend` |
| `expected Tensor<_, 3> found Tensor<_, 2>` | Rank mismatch | Check operation — some ops change rank (sum, squeeze) |
| `cannot move out of borrowed content` | Tensor consumed by operation | Use `.clone()` before the consuming op |
| Slow test execution | Using wgpu backend in tests | Use burn-ndarray for unit tests |

</troubleshooting>

<references>
— crates/jepa-core/src/types.rs: Tensor wrapper patterns (Representation, Energy)
— Cargo.toml: Workspace burn dependency configuration
</references>
