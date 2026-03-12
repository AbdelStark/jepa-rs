//! # jepa-world
//!
//! World model primitives for JEPA-based planning and control.
//!
//! Implements RFC-009 (Action-Conditioned World Model) and
//! RFC-010 (Hierarchical JEPA / H-JEPA).
//!
//! ## Modules
//! - [`action`] — Action types and action-conditioned prediction trait
//! - [`planner`] — World model rollout and plan evaluation
//! - [`hierarchy`] — H-JEPA multi-scale hierarchy

pub mod action;
pub mod hierarchy;
pub mod memory;
pub mod planner;

pub use action::{Action, ActionConditionedPredictor};
pub use hierarchy::{HierarchicalJepa, JepaLevel};
pub use memory::ShortTermMemory;
pub use planner::{
    CostFunction, L2Cost, PlanResult, RandomShootingConfig, RandomShootingPlanner, WorldModel,
};
