//! # jepa-world
//!
//! World model primitives for JEPA-based planning and control.
//!
//! This crate bridges the gap between learned representations (from
//! [`jepa_core`] / `jepa-vision`) and decision-making. It provides:
//!
//! - **Action-conditioned dynamics** — predict next-state representations
//!   given an action, enabling model-based rollout (RFC-009).
//! - **Hierarchical JEPA (H-JEPA)** — stack multiple JEPA levels at
//!   different temporal/spatial scales (RFC-010).
//! - **Short-term memory** — bounded FIFO buffer of recent states for
//!   multi-step planning context.
//! - **Planning** — [`RandomShootingPlanner`] (CEM) evaluates candidate
//!   action sequences against a learned world model and cost function.
//!
//! ```text
//!                  ┌───────────────┐
//! observation ────►│   Encoder     │──► s_t
//!                  └───────────────┘     │
//!                                        ▼
//!                  ┌───────────────┐   ┌──────────────────────┐
//!          a_t ───►│  Dynamics     │──►│  s_{t+1} ... s_{t+H} │──► CostFunction ──► plan
//!                  │  (World Model)│   └──────────────────────┘
//!                  └───────────────┘
//! ```
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`action`] | [`Action`] wrapper, [`ActionConditionedPredictor`] trait |
//! | [`planner`] | [`WorldModel`], [`RandomShootingPlanner`] (CEM), [`CostFunction`] trait, [`L2Cost`] |
//! | [`hierarchy`] | [`HierarchicalJepa`], [`JepaLevel`] — multi-scale H-JEPA |
//! | [`memory`] | [`ShortTermMemory`] — bounded ring buffer of recent states |

pub mod action;
pub mod hierarchy;
pub mod memory;
pub mod planner;

pub use action::{Action, ActionConditionedPredictor};
pub use hierarchy::{HierarchicalJepa, HierarchyError, JepaLevel};
pub use memory::{MemoryError, ShortTermMemory};
pub use planner::{
    CostFunction, L2Cost, PlanResult, PlanningError, RandomShootingConfig, RandomShootingPlanner,
    WorldModel,
};
