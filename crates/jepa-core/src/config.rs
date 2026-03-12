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
        if self.embed_dim % self.num_heads != 0 {
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

impl JepaConfig {
    /// ViT-Base/16 preset: 12 layers, 768-d, 12 heads, patch 16x16.
    ///
    /// Standard ViT-B configuration used in many JEPA experiments.
    pub fn vit_base_16() -> Self {
        Self {
            embed_dim: 768,
            predictor_embed_dim: 384,
            num_encoder_layers: 12,
            num_predictor_layers: 6,
            num_heads: 12,
            patch_size: (16, 16),
            tubelet_size: (2, 16, 16),
            ema_momentum: 0.996,
        }
    }

    /// ViT-Large/16 preset: 24 layers, 1024-d, 16 heads, patch 16x16.
    ///
    /// Used by V-JEPA ViT-L/16 checkpoints.
    pub fn vit_large_16() -> Self {
        Self {
            embed_dim: 1024,
            predictor_embed_dim: 512,
            num_encoder_layers: 24,
            num_predictor_layers: 12,
            num_heads: 16,
            patch_size: (16, 16),
            tubelet_size: (2, 16, 16),
            ema_momentum: 0.996,
        }
    }

    /// ViT-Huge/14 preset: 32 layers, 1280-d, 16 heads, patch 14x14.
    ///
    /// Used by I-JEPA ViT-H/14 checkpoints.
    pub fn vit_huge_14() -> Self {
        Self {
            embed_dim: 1280,
            predictor_embed_dim: 640,
            num_encoder_layers: 32,
            num_predictor_layers: 12,
            num_heads: 16,
            patch_size: (14, 14),
            tubelet_size: (2, 14, 14),
            ema_momentum: 0.996,
        }
    }

    /// ViT-giant/14 preset: 40 layers, 1408-d, 16 heads, patch 14x14.
    ///
    /// Used by V-JEPA 2 ViT-g checkpoints.
    pub fn vit_giant_14() -> Self {
        Self {
            embed_dim: 1408,
            predictor_embed_dim: 704,
            num_encoder_layers: 40,
            num_predictor_layers: 12,
            num_heads: 16,
            patch_size: (14, 14),
            tubelet_size: (2, 14, 14),
            ema_momentum: 0.996,
        }
    }
}

/// Builder for [`JepaConfig`] with chainable setters.
///
/// # Example
///
/// ```
/// use jepa_core::config::JepaConfigBuilder;
///
/// let config = JepaConfigBuilder::new()
///     .embed_dim(512)
///     .num_heads(8)
///     .num_encoder_layers(12)
///     .build()
///     .expect("config should be valid");
/// assert_eq!(config.embed_dim, 512);
/// assert_eq!(config.head_dim(), 64);
/// ```
#[derive(Debug, Clone)]
pub struct JepaConfigBuilder {
    config: JepaConfig,
}

impl JepaConfigBuilder {
    /// Create a new builder starting from the default config.
    pub fn new() -> Self {
        Self {
            config: JepaConfig::default(),
        }
    }

    /// Create a builder starting from a named preset.
    pub fn from_preset(config: JepaConfig) -> Self {
        Self { config }
    }

    /// Set the encoder embedding dimension.
    pub fn embed_dim(mut self, dim: usize) -> Self {
        self.config.embed_dim = dim;
        self
    }

    /// Set the predictor embedding dimension.
    pub fn predictor_embed_dim(mut self, dim: usize) -> Self {
        self.config.predictor_embed_dim = dim;
        self
    }

    /// Set the number of encoder transformer layers.
    pub fn num_encoder_layers(mut self, n: usize) -> Self {
        self.config.num_encoder_layers = n;
        self
    }

    /// Set the number of predictor transformer layers.
    pub fn num_predictor_layers(mut self, n: usize) -> Self {
        self.config.num_predictor_layers = n;
        self
    }

    /// Set the number of attention heads.
    pub fn num_heads(mut self, n: usize) -> Self {
        self.config.num_heads = n;
        self
    }

    /// Set the image patch size `(height, width)`.
    pub fn patch_size(mut self, h: usize, w: usize) -> Self {
        self.config.patch_size = (h, w);
        self
    }

    /// Set the video tubelet size `(temporal, height, width)`.
    pub fn tubelet_size(mut self, t: usize, h: usize, w: usize) -> Self {
        self.config.tubelet_size = (t, h, w);
        self
    }

    /// Set the EMA momentum.
    pub fn ema_momentum(mut self, m: f64) -> Self {
        self.config.ema_momentum = m;
        self
    }

    /// Build and validate the config.
    ///
    /// Returns `Err(ConfigError)` if validation fails.
    pub fn build(self) -> Result<JepaConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for JepaConfigBuilder {
    fn default() -> Self {
        Self::new()
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

    // --- Preset tests ---

    #[test]
    fn test_vit_base_16_is_valid() {
        let config = JepaConfig::vit_base_16();
        assert!(config.validate().is_ok());
        assert_eq!(config.embed_dim, 768);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_vit_large_16_is_valid() {
        let config = JepaConfig::vit_large_16();
        assert!(config.validate().is_ok());
        assert_eq!(config.embed_dim, 1024);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_vit_huge_14_is_valid() {
        let config = JepaConfig::vit_huge_14();
        assert!(config.validate().is_ok());
        assert_eq!(config.embed_dim, 1280);
        assert_eq!(config.num_encoder_layers, 32);
        assert_eq!(config.patch_size, (14, 14));
    }

    #[test]
    fn test_vit_giant_14_is_valid() {
        let config = JepaConfig::vit_giant_14();
        assert!(config.validate().is_ok());
        assert_eq!(config.embed_dim, 1408);
        assert_eq!(config.num_encoder_layers, 40);
    }

    // --- Builder tests ---

    #[test]
    fn test_builder_default_is_valid() {
        let config = JepaConfigBuilder::new().build().unwrap();
        assert_eq!(config.embed_dim, 256);
    }

    #[test]
    fn test_builder_custom_embed_dim() {
        let config = JepaConfigBuilder::new()
            .embed_dim(512)
            .num_heads(8)
            .build()
            .unwrap();
        assert_eq!(config.embed_dim, 512);
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_builder_from_preset() {
        let config = JepaConfigBuilder::from_preset(JepaConfig::vit_huge_14())
            .ema_momentum(0.999)
            .build()
            .unwrap();
        assert_eq!(config.embed_dim, 1280);
        assert!((config.ema_momentum - 0.999).abs() < 1e-10);
    }

    #[test]
    fn test_builder_validates_on_build() {
        let result = JepaConfigBuilder::new()
            .embed_dim(255) // not divisible by 8 heads
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_all_setters() {
        let config = JepaConfigBuilder::new()
            .embed_dim(384)
            .predictor_embed_dim(192)
            .num_encoder_layers(6)
            .num_predictor_layers(3)
            .num_heads(6)
            .patch_size(8, 8)
            .tubelet_size(4, 8, 8)
            .ema_momentum(0.999)
            .build()
            .unwrap();
        assert_eq!(config.embed_dim, 384);
        assert_eq!(config.predictor_embed_dim, 192);
        assert_eq!(config.num_encoder_layers, 6);
        assert_eq!(config.num_predictor_layers, 3);
        assert_eq!(config.num_heads, 6);
        assert_eq!(config.patch_size, (8, 8));
        assert_eq!(config.tubelet_size, (4, 8, 8));
        assert!((config.ema_momentum - 0.999).abs() < 1e-10);
    }
}
