//! Configuration types for JEPA architecture.
//!
//! Implements RFC-001 (Core Tensor Abstractions) — configuration component.

use serde::{Deserialize, Serialize};

/// Configuration for JEPA architecture dimensions.
///
/// Specifies the hyperparameters that define the shape and size of
/// encoder, predictor, and training components.
///
/// # Example
/// ```
/// use jepa_core::config::JepaConfig;
///
/// let config = JepaConfig {
///     embed_dim: 256,
///     predictor_embed_dim: 128,
///     num_encoder_layers: 12,
///     num_predictor_layers: 6,
///     num_heads: 8,
///     patch_size: (16, 16),
///     tubelet_size: (2, 16, 16),
///     ema_momentum: 0.996,
/// };
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JepaConfig {
    /// Embedding dimension of the encoder output.
    pub embed_dim: usize,
    /// Embedding dimension of the predictor (can be smaller than encoder).
    pub predictor_embed_dim: usize,
    /// Number of transformer layers in the encoder.
    pub num_encoder_layers: usize,
    /// Number of transformer layers in the predictor.
    pub num_predictor_layers: usize,
    /// Number of attention heads (must divide embed_dim evenly).
    pub num_heads: usize,
    /// Patch size for images `(height, width)`.
    pub patch_size: (usize, usize),
    /// Tubelet size for video `(temporal, height, width)`.
    pub tubelet_size: (usize, usize, usize),
    /// EMA momentum for target encoder updates. Range: `[0.0, 1.0]`.
    pub ema_momentum: f64,
}

/// Errors from config validation.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("embed_dim must be positive, got {0}")]
    ZeroEmbedDim(usize),
    #[error("predictor_embed_dim must be positive, got {0}")]
    ZeroPredictorEmbedDim(usize),
    #[error("num_encoder_layers must be positive, got {0}")]
    ZeroEncoderLayers(usize),
    #[error("num_predictor_layers must be positive, got {0}")]
    ZeroPredictorLayers(usize),
    #[error("num_heads must be positive, got {0}")]
    ZeroHeads(usize),
    #[error("embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")]
    HeadDimMismatch { embed_dim: usize, num_heads: usize },
    #[error("patch_size dimensions must be positive, got ({0}, {1})")]
    ZeroPatchSize(usize, usize),
    #[error("tubelet_size dimensions must be positive, got ({0}, {1}, {2})")]
    ZeroTubeletSize(usize, usize, usize),
    #[error("ema_momentum must be in [0.0, 1.0], got {0}")]
    InvalidMomentum(f64),
}

impl JepaConfig {
    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.embed_dim == 0 {
            return Err(ConfigError::ZeroEmbedDim(self.embed_dim));
        }
        if self.predictor_embed_dim == 0 {
            return Err(ConfigError::ZeroPredictorEmbedDim(self.predictor_embed_dim));
        }
        if self.num_encoder_layers == 0 {
            return Err(ConfigError::ZeroEncoderLayers(self.num_encoder_layers));
        }
        if self.num_predictor_layers == 0 {
            return Err(ConfigError::ZeroPredictorLayers(self.num_predictor_layers));
        }
        if self.num_heads == 0 {
            return Err(ConfigError::ZeroHeads(self.num_heads));
        }
        if !self.embed_dim.is_multiple_of(self.num_heads) {
            return Err(ConfigError::HeadDimMismatch {
                embed_dim: self.embed_dim,
                num_heads: self.num_heads,
            });
        }
        if self.patch_size.0 == 0 || self.patch_size.1 == 0 {
            return Err(ConfigError::ZeroPatchSize(
                self.patch_size.0,
                self.patch_size.1,
            ));
        }
        if self.tubelet_size.0 == 0 || self.tubelet_size.1 == 0 || self.tubelet_size.2 == 0 {
            return Err(ConfigError::ZeroTubeletSize(
                self.tubelet_size.0,
                self.tubelet_size.1,
                self.tubelet_size.2,
            ));
        }
        if !(0.0..=1.0).contains(&self.ema_momentum) {
            return Err(ConfigError::InvalidMomentum(self.ema_momentum));
        }
        Ok(())
    }

    /// Head dimension: `embed_dim / num_heads`.
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }
}

impl Default for JepaConfig {
    fn default() -> Self {
        Self {
            embed_dim: 256,
            predictor_embed_dim: 128,
            num_encoder_layers: 12,
            num_predictor_layers: 6,
            num_heads: 8,
            patch_size: (16, 16),
            tubelet_size: (2, 16, 16),
            ema_momentum: 0.996,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = JepaConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_head_dim() {
        let config = JepaConfig::default();
        assert_eq!(config.head_dim(), 32); // 256 / 8
    }

    #[test]
    fn test_zero_embed_dim_rejected() {
        let config = JepaConfig {
            embed_dim: 0,
            ..JepaConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::ZeroEmbedDim(0))
        ));
    }

    #[test]
    fn test_head_dim_mismatch_rejected() {
        let config = JepaConfig {
            embed_dim: 255,
            ..JepaConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::HeadDimMismatch { .. })
        ));
    }

    #[test]
    fn test_invalid_momentum_rejected() {
        let config = JepaConfig {
            ema_momentum: 1.5,
            ..JepaConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidMomentum(_))
        ));
    }

    #[test]
    fn test_negative_momentum_rejected() {
        let config = JepaConfig {
            ema_momentum: -0.1,
            ..JepaConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidMomentum(_))
        ));
    }

    #[test]
    fn test_zero_patch_size_rejected() {
        let config = JepaConfig {
            patch_size: (0, 16),
            ..JepaConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(ConfigError::ZeroPatchSize(0, 16))
        ));
    }

    #[test]
    fn test_config_serialization_roundtrip() {
        let config = JepaConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: JepaConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.embed_dim, config.embed_dim);
        assert_eq!(deserialized.num_heads, config.num_heads);
    }
}
