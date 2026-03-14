//! # jepa-world
//!
//! World model primitives for JEPA-based planning and control.
//!
//! ```text
//! observation ──► Encoder ──► s_t ──► Dynamics(s_t, a_t) ──► s_{t+1}..s_{t+H} ──► Cost ──► plan
//! ```
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`action`] | [`Action`] wrapper, [`ActionConditionedPredictor`] trait |
//! | [`planner`] | [`WorldModel`], [`RandomShootingPlanner`] (CEM), [`CostFunction`], [`L2Cost`] |
//! | [`hierarchy`] | [`HierarchicalJepa`], [`JepaLevel`] — multi-scale H-JEPA |
//! | [`memory`] | [`ShortTermMemory`] — bounded ring buffer of recent states |
//! | [`object_dynamics`] | [`ObjectDynamicsPredictor`] — transformer dynamics for C-JEPA planning |

pub mod action;
pub mod hierarchy;
pub mod memory;
pub mod object_dynamics;
pub mod planner;

pub use action::{Action, ActionConditionedPredictor};
pub use hierarchy::{HierarchicalJepa, HierarchyError, JepaLevel};
pub use memory::{MemoryError, ShortTermMemory};
pub use object_dynamics::{ObjectDynamicsPredictor, ObjectDynamicsPredictorConfig};
pub use planner::{
    CostFunction, L2Cost, PlanResult, PlanningError, RandomShootingConfig, RandomShootingPlanner,
    WorldModel,
};
