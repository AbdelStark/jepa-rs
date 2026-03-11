//! # jepa-core
//!
//! Core traits and abstractions for the Joint Embedding Predictive Architecture (JEPA).
//!
//! JEPA is a self-supervised learning architecture that predicts in representation space
//! rather than pixel space. Proposed by Yann LeCun in "A Path Towards Autonomous Machine
//! Intelligence" (2022), JEPA is the foundation for world models that understand the
//! physical world.
//!
//! This crate provides the foundational building blocks:
//! - [`Encoder`] trait for mapping inputs to representations
//! - [`Predictor`] trait for predicting target representations from context
//! - [`EnergyFn`] trait for measuring prediction quality
//! - [`MaskingStrategy`] trait for generating context/target splits
//! - [`CollapseRegularizer`] trait for preventing representational collapse
//! - [`Ema`] for exponential moving average target encoder updates

pub mod collapse;
pub mod config;
pub mod ema;
pub mod encoder;
pub mod energy;
pub mod masking;
pub mod predictor;
pub mod types;

pub use collapse::CollapseRegularizer;
pub use config::JepaConfig;
pub use ema::Ema;
pub use encoder::Encoder;
pub use energy::EnergyFn;
pub use masking::MaskingStrategy;
pub use predictor::Predictor;
pub use types::{Energy, MaskSpec, Representation};
