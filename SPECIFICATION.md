# jepa-rs: JEPA Primitives in Rust

## Specification and RFC Archive

**Version:** 0.1.0-draft
**Author:** Abdel Bakhta (@AbdelStark)
**Date:** March 2026
**License:** Apache-2.0 / MIT dual license

---

## Abstract

`jepa-rs` is a production-grade Rust library implementing the core primitives of the Joint Embedding Predictive Architecture (JEPA) as proposed by Yann LeCun. It provides the foundational building blocks for constructing world models that predict in representation space rather than pixel space.

This is the first JEPA implementation in Rust. All existing implementations are in Python/PyTorch (Meta's `ijepa`, `jepa`, `eb_jepa`, `jepa-wms`). A Rust implementation enables deployment in safety-critical environments, embedded systems, and anywhere Python/PyTorch overhead is unacceptable.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Architecture Overview](#2-architecture-overview)
3. [RFC-001: Core Tensor Abstractions](#rfc-001)
4. [RFC-002: Encoder Module](#rfc-002)
5. [RFC-003: Predictor Module](#rfc-003)
6. [RFC-004: Energy Functions](#rfc-004)
7. [RFC-005: Masking Strategies](#rfc-005)
8. [RFC-006: Collapse Prevention (VICReg)](#rfc-006)
9. [RFC-007: EMA Target Encoder](#rfc-007)
10. [RFC-008: Training Loop](#rfc-008)
11. [RFC-009: Action-Conditioned World Model](#rfc-009)
12. [RFC-010: Hierarchical JEPA (H-JEPA)](#rfc-010)
13. [Test Strategy](#test-strategy)
14. [BDD Specifications](#bdd-specifications)
15. [Differential Testing](#differential-testing)
16. [Crate Structure](#crate-structure)
17. [Reference Implementations](#reference-implementations)
18. [Bibliography](#bibliography)

---

## 1. Motivation

### Why Rust for JEPA?

1. **Safety-critical deployment.** AMI targets healthcare, robotics, and industrial automation. These domains need memory safety guarantees that Python cannot provide. Rust's ownership model eliminates entire classes of bugs at compile time.

2. **Inference on constrained hardware.** World models will run on wearable devices, robots, and edge hardware. Rust compiles to bare metal with no runtime overhead. A Rust JEPA inference engine can run on an ARM Cortex-M where Python cannot.

3. **Deterministic execution.** For verifiable computation and auditable AI (the ZK + safety thesis), deterministic execution is a prerequisite. Rust's lack of garbage collector and explicit memory model make execution fully deterministic.

4. **Production infrastructure.** Data pipelines, model serving, video preprocessing at scale. Rust's performance and concurrency model make it ideal for the infrastructure around world model training.

5. **No existing implementation.** As of March 2026, zero Rust implementations of JEPA exist. Every implementation (I-JEPA, V-JEPA, V-JEPA 2, EB-JEPA) is PyTorch-only.

### Scope

This library provides primitives, not a complete training framework. It is to JEPA what `ring` is to cryptography: the low-level building blocks that higher-level systems compose.

**In scope:**
- JEPA core architecture (encoders, predictors, energy functions)
- Masking strategies (image, video, spatiotemporal)
- Collapse prevention methods (VICReg, Barlow Twins, EMA)
- Action conditioning for world models
- Hierarchical JEPA (H-JEPA) stacking
- Weight loading from PyTorch checkpoints (safetensors format)
- Deterministic inference for verifiable computation

**Out of scope (for v0.1):**
- Full distributed training framework
- GPU kernel authoring (defer to burn/candle backends)
- Video decoding (use external crates)
- Dataset management

### Backend Strategy

`jepa-rs` is backend-agnostic via the `burn` framework's `Backend` trait. This allows:
- CPU inference via `burn-ndarray`
- GPU training/inference via `burn-wgpu` or `burn-cuda`
- WASM deployment via `burn-ndarray` + wasm32 target
- Future: custom backends for specific hardware

The `candle` crate from HuggingFace is an alternative backend option, particularly for loading pre-trained PyTorch models.

---

## 2. Architecture Overview

```
jepa-rs/
├── Cargo.toml
├── crates/
│   ├── jepa-core/           # Core traits and abstractions
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── encoder.rs    # Encoder trait and implementations
│   │   │   ├── predictor.rs  # Predictor trait and implementations
│   │   │   ├── energy.rs     # Energy functions (L2, cosine, etc.)
│   │   │   ├── masking.rs    # Masking strategies
│   │   │   ├── collapse.rs   # Collapse prevention (VICReg, BarlowTwins)
│   │   │   ├── ema.rs        # Exponential Moving Average
│   │   │   └── config.rs     # Configuration types
│   │   └── Cargo.toml
│   ├── jepa-vision/          # Vision-specific implementations (ViT encoder)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── vit.rs        # Vision Transformer
│   │   │   ├── patch.rs      # Image/video patchification
│   │   │   ├── rope.rs       # Rotary Position Embedding (2D/3D)
│   │   │   ├── image.rs      # I-JEPA specific logic
│   │   │   └── video.rs      # V-JEPA specific logic (tubelets)
│   │   └── Cargo.toml
│   ├── jepa-world/           # World model primitives
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── action.rs     # Action conditioning
│   │   │   ├── planner.rs    # Model-based planning
│   │   │   ├── hierarchy.rs  # H-JEPA stacking
│   │   │   └── memory.rs     # Short-term memory module
│   │   └── Cargo.toml
│   ├── jepa-train/           # Training utilities
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── loop.rs       # Training loop
│   │   │   ├── schedule.rs   # Learning rate schedules
│   │   │   └── checkpoint.rs # Checkpoint save/load
│   │   └── Cargo.toml
│   └── jepa-compat/          # PyTorch compatibility
│       ├── src/
│       │   ├── lib.rs
│       │   ├── safetensors.rs # Load safetensors weights
│       │   └── onnx.rs       # ONNX model import
│       └── Cargo.toml
├── tests/
│   ├── differential/         # Differential tests against Python reference
│   ├── bdd/                  # BDD feature tests
│   ├── fixtures/             # Test vectors from reference implementations
│   └── fuzz/                 # Fuzz targets
├── benches/                  # Criterion benchmarks
├── examples/
│   ├── ijepa_cifar10.rs      # I-JEPA on CIFAR-10
│   ├── vjepa_simple.rs       # V-JEPA on simple video
│   └── world_model_2rooms.rs # World model planning in Two Rooms env
└── specs/
    ├── rfcs/                 # This document
    └── gherkin/              # BDD feature files
```

---

## RFC-001: Core Tensor Abstractions

### Summary
Define the foundational traits that all JEPA components implement, remaining backend-agnostic.

### Design

```rust
use burn::tensor::{backend::Backend, Tensor};

/// A representation produced by an encoder.
/// Wraps a tensor with semantic meaning.
pub struct Representation<B: Backend> {
    /// The embedding tensor. Shape: [batch, seq_len, embed_dim]
    pub embeddings: Tensor<B, 3>,
    /// Optional mask indicating which positions are valid
    pub mask: Option<Tensor<B, 2>>,
}

/// Energy scalar measuring compatibility between two representations.
pub struct Energy<B: Backend> {
    pub value: Tensor<B, 1>,
}

/// Configuration for JEPA architecture dimensions.
#[derive(Debug, Clone)]
pub struct JepaConfig {
    pub embed_dim: usize,
    pub predictor_embed_dim: usize,
    pub num_encoder_layers: usize,
    pub num_predictor_layers: usize,
    pub num_heads: usize,
    pub patch_size: (usize, usize),      // For images
    pub tubelet_size: (usize, usize, usize), // For video (t, h, w)
    pub ema_momentum: f64,
}
```

### Rationale
Keeping representations as explicit types (not bare tensors) enables type-safe composition and makes the API self-documenting. An `Energy` is not interchangeable with a `Representation` at the type level.

---

## RFC-002: Encoder Module

### Summary
Define the encoder trait and provide a Vision Transformer (ViT) implementation.

### Design

```rust
/// Trait for JEPA encoders.
/// An encoder maps raw input to a representation.
pub trait Encoder<B: Backend>: burn::module::Module<B> {
    type Input;
    
    /// Encode input into a representation.
    fn encode(&self, input: &Self::Input) -> Representation<B>;
    
    /// Get the output embedding dimension.
    fn embed_dim(&self) -> usize;
}

/// Vision Transformer encoder for images.
pub struct VitEncoder<B: Backend> {
    patch_embed: PatchEmbedding<B>,
    positional_encoding: RotaryPositionalEncoding<B>,
    transformer_blocks: Vec<TransformerBlock<B>>,
    layer_norm: LayerNorm<B>,
    config: VitConfig,
}

/// Vision Transformer encoder for video (3D tubelets).
pub struct VitVideoEncoder<B: Backend> {
    tubelet_embed: TubeletEmbedding<B>,
    positional_encoding: RotaryPositionalEncoding3D<B>,
    transformer_blocks: Vec<TransformerBlock<B>>,
    layer_norm: LayerNorm<B>,
    config: VitVideoConfig,
}
```

### Context Encoder vs Target Encoder
In JEPA, the context (X) and target (Y) encoders can have different architectures but typically share the same structure. The target encoder's weights are updated via EMA (see RFC-007), not via gradient descent.

```rust
pub struct JepaEncoderPair<B: Backend, E: Encoder<B>> {
    pub context_encoder: E,
    pub target_encoder: E, // Updated via EMA, not gradients
}
```

---

## RFC-003: Predictor Module

### Summary
The predictor maps a context representation + latent variable to a predicted target representation.

### Design

```rust
/// Trait for JEPA predictors.
pub trait Predictor<B: Backend>: burn::module::Module<B> {
    /// Predict target representation from context representation.
    /// 
    /// # Arguments
    /// * `context` - Representation from the context encoder
    /// * `target_positions` - Positions of the target tokens to predict
    /// * `latent` - Optional latent variable z for stochastic predictions
    fn predict(
        &self,
        context: &Representation<B>,
        target_positions: &Tensor<B, 2>,  // [batch, num_targets]
        latent: Option<&Tensor<B, 2>>,
    ) -> Representation<B>;
}

/// Standard transformer-based predictor.
pub struct TransformerPredictor<B: Backend> {
    /// Learnable prediction tokens (one per target position)
    prediction_tokens: Tensor<B, 2>,
    /// Positional embedding for prediction targets
    positional_embedding: RotaryPositionalEncoding<B>,
    /// Cross-attention from prediction tokens to context representations
    transformer_blocks: Vec<CrossAttentionBlock<B>>,
    /// Final projection
    projection: Linear<B>,
    config: PredictorConfig,
}
```

### Key Design Decision
The predictor uses **cross-attention** from learnable prediction tokens to the context encoder's output. This is how I-JEPA and V-JEPA work: prediction tokens attend to the visible (unmasked) representation to predict the masked representation. The prediction tokens are initialized with the target positional embeddings, giving the predictor information about WHERE to predict.

---

## RFC-004: Energy Functions

### Summary
Energy functions measure the distance between predicted and actual target representations.

### Design

```rust
/// Trait for energy functions.
pub trait EnergyFn<B: Backend> {
    /// Compute energy (distance) between predicted and actual representations.
    /// Lower energy = better prediction = more compatible pair.
    fn compute(
        &self,
        predicted: &Representation<B>,
        actual: &Representation<B>,
    ) -> Energy<B>;
}

/// L2 distance in representation space (used by I-JEPA, V-JEPA).
pub struct L2Energy;

impl<B: Backend> EnergyFn<B> for L2Energy {
    fn compute(&self, predicted: &Representation<B>, actual: &Representation<B>) -> Energy<B> {
        // Mean squared error between predicted and actual representations
        // averaged over the embedding dimension
        let diff = predicted.embeddings.clone() - actual.embeddings.clone();
        let squared = diff.clone() * diff;
        let mean = squared.mean();
        Energy { value: mean }
    }
}

/// Cosine similarity energy (alternative).
pub struct CosineEnergy;

/// Smooth L1 energy (Huber loss variant).
pub struct SmoothL1Energy {
    pub beta: f64,
}
```

---

## RFC-005: Masking Strategies

### Summary
Masking strategies determine which parts of the input are visible (context) and which are hidden (targets). This is the most critical design decision in JEPA, as the masking strategy determines what the model learns.

### Design

```rust
/// A mask specification describing which tokens/patches are visible vs hidden.
#[derive(Debug, Clone)]
pub struct MaskSpec {
    /// Indices of context (visible) tokens
    pub context_indices: Vec<usize>,
    /// Indices of target (hidden) tokens
    pub target_indices: Vec<usize>,
    /// Total number of tokens
    pub total_tokens: usize,
}

/// Trait for masking strategies.
pub trait MaskingStrategy {
    /// Generate a mask for a given input shape.
    fn generate_mask(&self, shape: &InputShape, rng: &mut impl Rng) -> MaskSpec;
}

/// Block masking for images (I-JEPA style).
/// Masks one or more contiguous rectangular blocks.
pub struct BlockMasking {
    /// Number of target blocks to mask
    pub num_targets: usize,
    /// Target block scale range (fraction of image area)
    pub target_scale: (f64, f64),
    /// Target block aspect ratio range
    pub target_aspect_ratio: (f64, f64),
    /// Context block scale range
    pub context_scale: (f64, f64),
}

/// Spatiotemporal masking for video (V-JEPA style).
/// Masks contiguous 3D regions in space and time.
pub struct SpatiotemporalMasking {
    /// Number of target tubes to mask
    pub num_targets: usize,
    /// Temporal extent of each tube (in frames)
    pub temporal_extent: (usize, usize),
    /// Spatial scale of each tube
    pub spatial_scale: (f64, f64),
    /// Minimum gap between tubes
    pub min_gap: usize,
}

/// Multi-block masking (V-JEPA 2 style).
/// Masks multiple blocks with specific constraints on coverage.
pub struct MultiBlockMasking {
    /// Target masking ratio (fraction of tokens masked)
    pub mask_ratio: f64,
    /// Number of mask blocks
    pub num_blocks: usize,
    /// Minimum spatial extent per block
    pub min_spatial_extent: usize,
    /// Minimum temporal extent per block  
    pub min_temporal_extent: usize,
}
```

### Test Vectors
Masking strategies must be tested against the reference Python implementations. The `facebookresearch/ijepa` repo contains the `MultiBlockMaskCollator` class that serves as the ground truth for I-JEPA masking. The `facebookresearch/jepa` repo contains V-JEPA masking.

---

## RFC-006: Collapse Prevention (VICReg)

### Summary
Without collapse prevention, JEPA training produces trivial solutions (all representations collapse to a constant). VICReg (Variance-Invariance-Covariance Regularization) is the primary method used by JEPA-family models.

### Design

```rust
/// VICReg regularization loss.
/// Prevents collapse by enforcing:
/// 1. Variance: each embedding dimension has high variance across the batch
/// 2. Invariance: representations of positive pairs are similar
/// 3. Covariance: different embedding dimensions capture different information
pub struct VICReg {
    /// Weight for the invariance term
    pub lambda_inv: f64,  // default: 25.0
    /// Weight for the variance term
    pub mu_var: f64,       // default: 25.0
    /// Weight for the covariance term
    pub nu_cov: f64,       // default: 1.0
    /// Target standard deviation for variance term
    pub gamma: f64,        // default: 1.0
    /// Epsilon for numerical stability
    pub eps: f64,          // default: 1e-4
}

impl VICReg {
    /// Compute the VICReg loss.
    pub fn loss<B: Backend>(
        &self,
        z_a: &Tensor<B, 2>, // [batch, embed_dim] - representation A
        z_b: &Tensor<B, 2>, // [batch, embed_dim] - representation B
    ) -> VICRegLoss<B> {
        // Invariance: MSE between representations
        let inv_loss = mse(z_a, z_b);
        
        // Variance: hinge loss on std dev of each dimension
        let std_a = z_a.var(0).sqrt().clamp_min(self.eps);
        let std_b = z_b.var(0).sqrt().clamp_min(self.eps);
        let var_loss = relu(self.gamma - std_a).mean() 
                     + relu(self.gamma - std_b).mean();
        
        // Covariance: off-diagonal elements of covariance matrix should be zero
        let cov_a = covariance_matrix(z_a);
        let cov_b = covariance_matrix(z_b);
        let cov_loss = off_diagonal(cov_a).pow(2).sum() / embed_dim
                     + off_diagonal(cov_b).pow(2).sum() / embed_dim;
        
        VICRegLoss {
            invariance: inv_loss * self.lambda_inv,
            variance: var_loss * self.mu_var,
            covariance: cov_loss * self.nu_cov,
        }
    }
}

/// Barlow Twins regularization (alternative).
pub struct BarlowTwins {
    pub lambda_bt: f64, // default: 0.005
}
```

### Test Vectors for Collapse Detection

```rust
/// Property test: representations should NOT collapse.
/// After N training steps, verify:
/// 1. Variance of representations across batch > threshold
/// 2. Rank of representation matrix > threshold
/// 3. Different inputs produce different representations
fn test_no_collapse(model: &JepaModel, data: &[Input]) -> bool {
    let reps: Vec<Representation> = data.iter().map(|x| model.encode(x)).collect();
    let stacked = stack_representations(&reps);
    
    let variance = stacked.var(0).mean();  // Should be > 0.1
    let rank = numerical_rank(&stacked);   // Should be > embed_dim / 2
    
    variance > 0.1 && rank > embed_dim / 2
}
```

---

## RFC-007: EMA Target Encoder

### Summary
The target encoder in JEPA is not trained directly via gradient descent. Instead, its weights are an Exponential Moving Average (EMA) of the context encoder's weights. This asymmetry between the two encoders is critical for preventing collapse.

### Design

```rust
/// Exponential Moving Average weight updater.
pub struct EMA {
    /// Momentum parameter. Typical values: 0.996 to 0.9999
    /// Higher = slower update (target changes slowly)
    pub momentum: f64,
    /// Optional momentum schedule (increases during training)
    pub schedule: Option<MomentumSchedule>,
}

impl EMA {
    /// Update target parameters from online (context) parameters.
    /// target = momentum * target + (1 - momentum) * online
    pub fn update<B: Backend>(
        &self,
        target: &mut impl burn::module::Module<B>,
        online: &impl burn::module::Module<B>,
        step: usize,
    ) {
        let m = match &self.schedule {
            Some(s) => s.get_momentum(step),
            None => self.momentum,
        };
        // For each parameter pair:
        // target_param = m * target_param + (1 - m) * online_param
    }
}

/// Cosine momentum schedule (V-JEPA 2 style).
/// Momentum increases from base to 1.0 over training.
pub struct CosineMomentumSchedule {
    pub base_momentum: f64,  // e.g., 0.996
    pub final_momentum: f64, // e.g., 1.0
    pub total_steps: usize,
}
```

---

## RFC-008: Training Loop

### Summary
The JEPA training loop orchestrates masking, encoding, prediction, and loss computation.

### Design (pseudocode)

```rust
pub struct JepaTrainer<B: Backend, E: Encoder<B>, P: Predictor<B>> {
    pub context_encoder: E,
    pub target_encoder: E,  // EMA copy
    pub predictor: P,
    pub masking: Box<dyn MaskingStrategy>,
    pub energy_fn: Box<dyn EnergyFn<B>>,
    pub collapse_prevention: VICReg,
    pub ema: EMA,
    pub optimizer: AdamW<B>,
}

impl JepaTrainer {
    pub fn train_step(&mut self, batch: &Batch<B>) -> TrainStepOutput<B> {
        // 1. Generate mask
        let mask = self.masking.generate_mask(&batch.shape(), &mut rng);
        
        // 2. Encode context (visible tokens) with context encoder
        //    Gradients flow through this encoder
        let context_input = apply_mask(&batch, &mask.context_indices);
        let context_repr = self.context_encoder.encode(&context_input);
        
        // 3. Encode targets (hidden tokens) with target encoder
        //    NO gradients (stop gradient / detach)
        let target_input = apply_mask(&batch, &mask.target_indices);
        let target_repr = no_grad(|| self.target_encoder.encode(&target_input));
        
        // 4. Predict target representations from context
        let predicted_repr = self.predictor.predict(
            &context_repr,
            &mask.target_positions(),
            None, // no latent for basic JEPA
        );
        
        // 5. Compute energy (prediction loss)
        let energy = self.energy_fn.compute(&predicted_repr, &target_repr);
        
        // 6. Compute collapse prevention loss
        let vicreg_loss = self.collapse_prevention.loss(
            &predicted_repr.embeddings,
            &target_repr.embeddings,
        );
        
        // 7. Total loss = energy + vicreg regularization
        let total_loss = energy.value + vicreg_loss.total();
        
        // 8. Backward pass (updates context_encoder and predictor only)
        total_loss.backward();
        self.optimizer.step();
        
        // 9. EMA update of target encoder
        self.ema.update(&mut self.target_encoder, &self.context_encoder, step);
        
        TrainStepOutput { energy, vicreg_loss, total_loss }
    }
}
```

---

## RFC-009: Action-Conditioned World Model

### Summary
Extend JEPA to predict future states given actions, enabling planning.

### Design

```rust
/// An action in the environment.
pub struct Action<B: Backend> {
    pub data: Tensor<B, 2>,  // [batch, action_dim]
}

/// Action-conditioned predictor.
/// Given current state representation + action, predicts next state representation.
pub trait ActionConditionedPredictor<B: Backend>: burn::module::Module<B> {
    fn predict_next_state(
        &self,
        current_state: &Representation<B>,
        action: &Action<B>,
    ) -> Representation<B>;
}

/// World model that can be used for planning.
pub struct WorldModel<B: Backend> {
    pub encoder: Box<dyn Encoder<B>>,
    pub dynamics: Box<dyn ActionConditionedPredictor<B>>,
    pub cost: Box<dyn CostFunction<B>>,
}

impl<B: Backend> WorldModel<B> {
    /// Simulate a sequence of actions and return predicted states.
    pub fn rollout(
        &self,
        initial_state: &Representation<B>,
        actions: &[Action<B>],
    ) -> Vec<Representation<B>> {
        let mut states = vec![initial_state.clone()];
        for action in actions {
            let next = self.dynamics.predict_next_state(states.last().unwrap(), action);
            states.push(next);
        }
        states
    }
    
    /// Evaluate a plan (sequence of actions) by computing total cost.
    pub fn evaluate_plan(
        &self,
        initial_state: &Representation<B>,
        actions: &[Action<B>],
        goal: &Representation<B>,
    ) -> Energy<B> {
        let trajectory = self.rollout(initial_state, actions);
        self.cost.total_cost(&trajectory, goal)
    }
}
```

---

## RFC-010: Hierarchical JEPA (H-JEPA)

### Summary
Stack multiple JEPA levels for multi-scale prediction.

### Design

```rust
/// A single level in the H-JEPA hierarchy.
pub struct JepaLevel<B: Backend> {
    pub encoder: Box<dyn Encoder<B>>,
    pub predictor: Box<dyn Predictor<B>>,
    /// Temporal abstraction factor (how many lower-level steps per higher-level step)
    pub temporal_stride: usize,
    /// Spatial pooling to reduce resolution between levels
    pub spatial_pool: Option<SpatialPooling<B>>,
}

/// Hierarchical JEPA with multiple abstraction levels.
pub struct HierarchicalJepa<B: Backend> {
    pub levels: Vec<JepaLevel<B>>,
}

impl<B: Backend> HierarchicalJepa<B> {
    /// Encode input at all levels of the hierarchy.
    pub fn encode_all_levels(&self, input: &Tensor<B, 5>) -> Vec<Representation<B>> {
        let mut representations = Vec::new();
        let mut current_input = input.clone();
        
        for level in &self.levels {
            let repr = level.encoder.encode(&current_input);
            representations.push(repr.clone());
            // Pool and stride for next level
            current_input = level.prepare_next_level_input(&repr);
        }
        
        representations
    }
    
    /// Predict at a specific level of abstraction.
    pub fn predict_at_level(
        &self,
        level_idx: usize,
        context: &Representation<B>,
        target_positions: &Tensor<B, 2>,
    ) -> Representation<B> {
        self.levels[level_idx].predictor.predict(context, target_positions, None)
    }
}
```

---

## Test Strategy

### Layer 1: Unit Tests (TDD)

Every function has corresponding unit tests written BEFORE implementation.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_energy_identical_representations() {
        // Energy between identical representations should be zero
        let repr = Representation::random([4, 16, 256]); // [batch, seq, embed]
        let energy = L2Energy.compute(&repr, &repr);
        assert!(energy.value.to_scalar::<f32>() < 1e-6);
    }
    
    #[test]
    fn test_l2_energy_different_representations() {
        // Energy between different representations should be positive
        let repr_a = Representation::random([4, 16, 256]);
        let repr_b = Representation::random([4, 16, 256]);
        let energy = L2Energy.compute(&repr_a, &repr_b);
        assert!(energy.value.to_scalar::<f32>() > 0.0);
    }
    
    #[test]
    fn test_vicreg_prevents_collapse() {
        // After applying VICReg loss, representations should have high variance
        let vicreg = VICReg::default();
        let z = Tensor::zeros([32, 256]); // Collapsed representation
        let loss = vicreg.loss(&z, &z);
        // Variance term should be high for collapsed representations
        assert!(loss.variance.to_scalar::<f32>() > 10.0);
    }

    #[test]
    fn test_ema_update_moves_toward_online() {
        let ema = EMA { momentum: 0.99, schedule: None };
        let online_weight = 1.0;
        let target_weight = 0.0;
        // After update: target = 0.99 * 0.0 + 0.01 * 1.0 = 0.01
        let result = ema.step(target_weight, online_weight);
        assert!((result - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_block_masking_coverage() {
        let masking = BlockMasking {
            num_targets: 4,
            target_scale: (0.15, 0.2),
            target_aspect_ratio: (0.75, 1.5),
            context_scale: (0.85, 1.0),
        };
        let shape = InputShape::Image { height: 14, width: 14 }; // 196 patches
        let mask = masking.generate_mask(&shape, &mut rng);
        
        // Context + target should cover all tokens
        assert_eq!(
            mask.context_indices.len() + mask.target_indices.len(),
            196
        );
        // No overlap
        let context_set: HashSet<_> = mask.context_indices.iter().collect();
        for t in &mask.target_indices {
            assert!(!context_set.contains(t));
        }
    }
}
```

### Layer 2: BDD Feature Tests (Gherkin)

```gherkin
# specs/gherkin/encoding.feature

Feature: JEPA Encoding
  As a developer using jepa-rs
  I want to encode inputs into representations
  So that I can make predictions in representation space

  Scenario: Encode a batch of images into representations
    Given a ViT encoder with embed_dim 256 and patch_size 16x16
    And a batch of 4 images of size 224x224x3
    When I encode the batch
    Then I should get representations of shape [4, 196, 256]
    And the representations should have non-zero variance across the batch

  Scenario: Context and target encoders produce compatible representations
    Given a JEPA encoder pair with shared architecture
    And the target encoder initialized as a copy of the context encoder
    When I encode the same image with both encoders
    Then the representations should be identical
    And the energy between them should be approximately zero

  Scenario: EMA update makes target encoder lag behind context encoder
    Given a JEPA encoder pair
    And the context encoder has been updated by gradient descent
    When I apply EMA update with momentum 0.99
    Then the target encoder weights should be closer to the context encoder
    And the target encoder weights should NOT equal the context encoder weights
```

```gherkin
# specs/gherkin/masking.feature

Feature: JEPA Masking Strategies
  As a developer training a JEPA model
  I want masking strategies that create meaningful prediction tasks
  So that the model learns useful representations

  Scenario: Block masking for images covers the correct area
    Given a block masking strategy with target_scale (0.15, 0.2)
    And an image tokenized into a 14x14 grid (196 patches)
    When I generate a mask
    Then between 15% and 20% of patches should be masked as targets
    And the masked patches should form contiguous rectangular blocks

  Scenario: Spatiotemporal masking for video spans space and time
    Given a spatiotemporal masking strategy for 16-frame video
    And video tokenized into 8x14x14 tubelets (1568 tokens)
    When I generate a mask
    Then each masked region should span at least 2 frames temporally
    And each masked region should span at least 4x4 patches spatially

  Scenario: Masking is stochastic across batches
    Given any masking strategy
    When I generate masks for two different batches
    Then the masks should be different with high probability
```

```gherkin
# specs/gherkin/world_model.feature

Feature: Action-Conditioned World Model
  As a developer building a planning agent
  I want to predict future states given actions
  So that I can plan action sequences to achieve goals

  Scenario: Rollout produces state trajectory
    Given a trained action-conditioned world model
    And an initial state representation
    And a sequence of 10 actions
    When I perform a rollout
    Then I should get 11 state representations (initial + 10 predicted)
    And each predicted state should have the same shape as the initial state

  Scenario: Plan evaluation computes cost
    Given a world model with a goal-reaching cost function
    And an initial state and a goal state
    And two candidate action sequences
    When I evaluate both plans
    Then the plan reaching closer to the goal should have lower cost
```

### Layer 3: Differential Testing

Compare jepa-rs outputs against the Python reference implementations.

```rust
/// Differential test framework.
/// Runs the same computation in both jepa-rs and the Python reference,
/// then compares outputs within a tolerance.

#[cfg(test)]
mod differential_tests {
    use std::process::Command;

    /// Generate test vectors from the Python reference implementation.
    /// This runs a Python script that:
    /// 1. Initializes I-JEPA with known random seed
    /// 2. Performs a forward pass on a known input
    /// 3. Saves the intermediate representations and outputs as .npz files
    fn generate_python_reference(test_name: &str) -> PathBuf {
        let output = Command::new("python")
            .arg("tests/differential/generate_reference.py")
            .arg("--test")
            .arg(test_name)
            .arg("--seed")
            .arg("42")
            .output()
            .expect("Failed to run Python reference");
        assert!(output.status.success(), "Python reference failed");
        PathBuf::from(format!("tests/fixtures/{}.npz", test_name))
    }

    #[test]
    fn test_vit_encoder_matches_python() {
        let reference = generate_python_reference("vit_encoder_forward");
        let (input, expected_output) = load_npz(&reference);
        
        let encoder = VitEncoder::from_reference_weights("tests/fixtures/vit_weights.safetensors");
        let output = encoder.encode(&input);
        
        assert_tensors_close(&output.embeddings, &expected_output, 1e-5);
    }

    #[test]
    fn test_vicreg_loss_matches_python() {
        let reference = generate_python_reference("vicreg_loss");
        let (z_a, z_b, expected_loss) = load_npz(&reference);
        
        let vicreg = VICReg::default();
        let loss = vicreg.loss(&z_a, &z_b);
        
        assert_scalar_close(loss.total(), expected_loss, 1e-4);
    }

    #[test]
    fn test_ema_update_matches_python() {
        let reference = generate_python_reference("ema_update");
        let (before_target, online, expected_after) = load_npz(&reference);
        
        let ema = EMA { momentum: 0.996, schedule: None };
        let after = ema.apply(&before_target, &online);
        
        assert_tensors_close(&after, &expected_after, 1e-6);
    }

    #[test]
    fn test_full_jepa_forward_matches_python() {
        let reference = generate_python_reference("ijepa_full_forward");
        let (input, mask, expected_energy) = load_npz(&reference);
        
        let model = IjepaModel::from_reference_checkpoint(
            "tests/fixtures/ijepa_checkpoint.safetensors"
        );
        let output = model.forward(&input, &mask);
        
        assert_scalar_close(output.energy, expected_energy, 1e-4);
    }
}
```

### Layer 4: Fuzz Testing

```rust
// tests/fuzz/fuzz_targets/fuzz_masking.rs

#![no_main]
use libfuzzer_sys::fuzz_target;
use jepa_core::masking::*;

fuzz_target!(|data: (u8, u8, u8, u64)| {
    let (h, w, num_targets, seed) = data;
    let h = (h as usize).max(4).min(64);
    let w = (w as usize).max(4).min(64);
    let num_targets = (num_targets as usize).max(1).min(8);
    
    let masking = BlockMasking {
        num_targets,
        target_scale: (0.1, 0.3),
        target_aspect_ratio: (0.5, 2.0),
        context_scale: (0.7, 1.0),
    };
    
    let shape = InputShape::Image { height: h, width: w };
    let mut rng = StdRng::seed_from_u64(seed);
    let mask = masking.generate_mask(&shape, &mut rng);
    
    // Invariants that must hold for any input:
    assert!(mask.context_indices.len() + mask.target_indices.len() <= h * w);
    assert!(!mask.target_indices.is_empty());
    assert!(!mask.context_indices.is_empty());
    // No duplicates
    let all: HashSet<_> = mask.context_indices.iter().chain(mask.target_indices.iter()).collect();
    assert_eq!(all.len(), mask.context_indices.len() + mask.target_indices.len());
});

// tests/fuzz/fuzz_targets/fuzz_energy.rs

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 { return; }
    // Parse arbitrary bytes as tensor data
    let floats: Vec<f32> = data.chunks(4)
        .filter_map(|c| c.try_into().ok().map(f32::from_le_bytes))
        .filter(|f| f.is_finite())
        .collect();
    if floats.len() < 8 { return; }
    
    let half = floats.len() / 2;
    let a = Tensor::from_slice(&floats[..half], [1, half]);
    let b = Tensor::from_slice(&floats[half..], [1, half.min(floats.len() - half)]);
    
    // Energy should never be NaN or negative
    if let Ok(energy) = L2Energy.compute_safe(&a, &b) {
        assert!(energy >= 0.0);
        assert!(!energy.is_nan());
    }
});
```

### Layer 5: Property-Based Testing

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn energy_is_symmetric(
        seed in 0u64..10000,
        batch in 1usize..8,
        dim in 16usize..512,
    ) {
        let a = Representation::random_with_seed([batch, 16, dim], seed);
        let b = Representation::random_with_seed([batch, 16, dim], seed + 1);
        
        let e_ab = L2Energy.compute(&a, &b);
        let e_ba = L2Energy.compute(&b, &a);
        
        prop_assert!((e_ab - e_ba).abs() < 1e-6);
    }
    
    #[test]
    fn energy_is_non_negative(
        seed in 0u64..10000,
        batch in 1usize..8,
        dim in 16usize..512,
    ) {
        let a = Representation::random_with_seed([batch, 16, dim], seed);
        let b = Representation::random_with_seed([batch, 16, dim], seed + 1);
        
        let energy = L2Energy.compute(&a, &b);
        prop_assert!(energy.value >= 0.0);
    }
    
    #[test]
    fn ema_converges_to_online(
        momentum in 0.9f64..0.9999,
        steps in 100usize..10000,
    ) {
        let ema = EMA { momentum, schedule: None };
        let online = 1.0f64;
        let mut target = 0.0f64;
        
        for _ in 0..steps {
            target = momentum * target + (1.0 - momentum) * online;
        }
        
        // After many steps, target should be close to online
        prop_assert!((target - online).abs() < 0.1);
    }
}
```

---

## Reference Implementations (for differential testing)

| Implementation | Language | Scope | Repository |
|---|---|---|---|
| **I-JEPA** | Python/PyTorch | Image JEPA | github.com/facebookresearch/ijepa |
| **V-JEPA / V-JEPA 2** | Python/PyTorch | Video JEPA | github.com/facebookresearch/jepa |
| **EB-JEPA** | Python/PyTorch | Educational library | github.com/facebookresearch/eb_jepa |
| **JEPA-WMs** | Python/PyTorch | World model planning | github.com/facebookresearch/jepa-wms |
| **JEPA demos** | Python/PyTorch | Research demos | github.com/yunusskeete/jepas |

### Key files for extracting test vectors

From `facebookresearch/ijepa`:
- `src/models/vision_transformer.py` - ViT encoder reference
- `src/masks/multiblock.py` - Block masking reference
- `src/helper.py` - EMA implementation reference

From `facebookresearch/eb_jepa`:
- `eb_jepa/models/jepa.py` - Clean JEPA reference
- `eb_jepa/losses/vicreg.py` - VICReg reference
- `examples/image_jepa/` - Complete I-JEPA training loop

From `facebookresearch/jepa`:
- `src/models/vision_transformer.py` - V-JEPA ViT with 3D RoPE
- `src/masks/` - Spatiotemporal masking
- `app/vjepa/train.py` - V-JEPA training loop

### Pre-trained checkpoints for loading tests

- I-JEPA ViT-H/14 (ImageNet): available from facebookresearch/ijepa releases
- V-JEPA ViT-L/16: available from facebookresearch/jepa releases
- V-JEPA 2 ViT-g: available from facebookresearch/jepa releases (safetensors format)

---

## Bibliography

1. LeCun, Y. (2022). "A Path Towards Autonomous Machine Intelligence." OpenReview.
2. Assran, M. et al. (2023). "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." CVPR.
3. Bardes, A. et al. (2024). "Revisiting Feature Prediction for Learning Visual Representations from Video." arXiv:2404.08471.
4. Assran, M. et al. (2025). "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning." arXiv:2506.09985.
5. Bardes, A. et al. (2022). "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." ICLR.
6. Zbontar, J. et al. (2021). "Barlow Twins: Self-Supervised Learning via Redundancy Reduction." ICML.
7. Terver, B. et al. (2026). "A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures." arXiv:2602.03604.
8. Terver, B. et al. (2025). "What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?" arXiv:2512.24497.
