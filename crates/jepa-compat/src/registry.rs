//! Registry of pretrained JEPA models from Facebook Research.
//!
//! Provides metadata, download URLs, and matching architecture configs for
//! official pretrained JEPA models. Use [`list_models`] to discover what
//! is available and [`find_model`] to look up a model by name.
//!
//! ## Available models
//!
//! | Model | Architecture | Resolution | Params | Family |
//! |-------|-------------|-----------|--------|--------|
//! | I-JEPA ViT-H/14 | ViT-Huge, patch 14 | 224×224 | 632 M | I-JEPA |
//! | I-JEPA ViT-H/16-448 | ViT-Huge, patch 16 | 448×448 | 632 M | I-JEPA |
//! | I-JEPA ViT-G/16 | ViT-Giant, patch 16 | 224×224 | 1.0 B | I-JEPA |
//! | V-JEPA ViT-L/16 | ViT-Large, patch 16 | 224×224 | 307 M | V-JEPA |
//! | V-JEPA ViT-H/16 | ViT-Huge, patch 16 | 224×224 | 632 M | V-JEPA |
//! | V-JEPA 2 ViT-g/14 | ViT-giant, patch 14 | 224×224 | 1.0 B | V-JEPA 2 |
//!
//! ## Example
//!
//! ```
//! use jepa_compat::registry::{PretrainedModel, list_models};
//!
//! for model in list_models() {
//!     println!("{}: {} params, {}", model.name, model.param_count_human(), model.architecture);
//! }
//! ```

/// A pretrained JEPA model available for download.
#[derive(Debug, Clone)]
pub struct PretrainedModel {
    /// Human-readable model name.
    pub name: &'static str,
    /// Model architecture description.
    pub architecture: &'static str,
    /// Model family (I-JEPA, V-JEPA, V-JEPA 2).
    pub family: ModelFamily,
    /// Input image resolution (height, width).
    pub resolution: (usize, usize),
    /// Patch size (height, width).
    pub patch_size: (usize, usize),
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// MLP hidden dimension.
    pub mlp_dim: usize,
    /// Approximate parameter count.
    pub num_params: u64,
    /// HuggingFace model hub URL (if available).
    pub huggingface_url: Option<&'static str>,
    /// Direct download URL for weights.
    pub weights_url: Option<&'static str>,
    /// GitHub repository with the reference implementation.
    pub source_repo: &'static str,
    /// Checkpoint format of the original weights.
    pub checkpoint_format: CheckpointFormat,
    /// Pretraining dataset.
    pub pretrained_on: &'static str,
}

impl PretrainedModel {
    /// Human-readable parameter count (e.g., "632M", "1.0B").
    pub fn param_count_human(&self) -> String {
        if self.num_params >= 1_000_000_000 {
            format!("{:.1}B", self.num_params as f64 / 1e9)
        } else {
            format!("{}M", self.num_params / 1_000_000)
        }
    }

    /// Number of patches for the model's input resolution.
    pub fn num_patches(&self) -> usize {
        let grid_h = self.resolution.0 / self.patch_size.0;
        let grid_w = self.resolution.1 / self.patch_size.1;
        grid_h * grid_w
    }
}

/// JEPA model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    /// Image JEPA (Assran et al., 2023).
    IJepa,
    /// Video JEPA (Bardes et al., 2024).
    VJepa,
    /// Video JEPA 2 (Bardes et al., 2025).
    VJepa2,
}

impl std::fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IJepa => write!(f, "I-JEPA"),
            Self::VJepa => write!(f, "V-JEPA"),
            Self::VJepa2 => write!(f, "V-JEPA 2"),
        }
    }
}

/// Format of the original pretrained checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointFormat {
    /// PyTorch state_dict (`.pth` or `.pt`).
    PyTorchStateDict,
    /// SafeTensors format (`.safetensors`).
    SafeTensors,
}

impl std::fmt::Display for CheckpointFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PyTorchStateDict => write!(f, "PyTorch (.pth)"),
            Self::SafeTensors => write!(f, "SafeTensors (.safetensors)"),
        }
    }
}

/// I-JEPA ViT-Huge/14 pretrained on ImageNet-1K at 224x224.
pub const IJEPA_VIT_H14_IN1K: PretrainedModel = PretrainedModel {
    name: "I-JEPA ViT-H/14 IN1K",
    architecture: "ViT-Huge/14",
    family: ModelFamily::IJepa,
    resolution: (224, 224),
    patch_size: (14, 14),
    embed_dim: 1280,
    num_layers: 32,
    num_heads: 16,
    mlp_dim: 5120,
    num_params: 632_000_000,
    huggingface_url: Some("https://huggingface.co/facebook/ijepa_vith14_1k"),
    weights_url: Some("https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar"),
    source_repo: "https://github.com/facebookresearch/ijepa",
    checkpoint_format: CheckpointFormat::PyTorchStateDict,
    pretrained_on: "ImageNet-1K",
};

/// I-JEPA ViT-Huge/16 pretrained on ImageNet-1K at 448x448.
pub const IJEPA_VIT_H16_448_IN1K: PretrainedModel = PretrainedModel {
    name: "I-JEPA ViT-H/16-448 IN1K",
    architecture: "ViT-Huge/16",
    family: ModelFamily::IJepa,
    resolution: (448, 448),
    patch_size: (16, 16),
    embed_dim: 1280,
    num_layers: 32,
    num_heads: 16,
    mlp_dim: 5120,
    num_params: 632_000_000,
    huggingface_url: Some("https://huggingface.co/facebook/ijepa_vith16_448"),
    weights_url: Some("https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16.448-300e.pth.tar"),
    source_repo: "https://github.com/facebookresearch/ijepa",
    checkpoint_format: CheckpointFormat::PyTorchStateDict,
    pretrained_on: "ImageNet-1K",
};

/// I-JEPA ViT-Giant/16 pretrained on ImageNet-22K at 224x224.
pub const IJEPA_VIT_G16_IN22K: PretrainedModel = PretrainedModel {
    name: "I-JEPA ViT-G/16 IN22K",
    architecture: "ViT-Giant/16",
    family: ModelFamily::IJepa,
    resolution: (224, 224),
    patch_size: (16, 16),
    embed_dim: 1408,
    num_layers: 40,
    num_heads: 16,
    mlp_dim: 6144,
    num_params: 1_012_000_000,
    huggingface_url: None,
    weights_url: Some("https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar"),
    source_repo: "https://github.com/facebookresearch/ijepa",
    checkpoint_format: CheckpointFormat::PyTorchStateDict,
    pretrained_on: "ImageNet-22K",
};

/// I-JEPA ViT-Huge/14 pretrained on ImageNet-22K at 224x224.
pub const IJEPA_VIT_H14_IN22K: PretrainedModel = PretrainedModel {
    name: "I-JEPA ViT-H/14 IN22K",
    architecture: "ViT-Huge/14",
    family: ModelFamily::IJepa,
    resolution: (224, 224),
    patch_size: (14, 14),
    embed_dim: 1280,
    num_layers: 32,
    num_heads: 16,
    mlp_dim: 5120,
    num_params: 632_000_000,
    huggingface_url: None,
    weights_url: Some("https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar"),
    source_repo: "https://github.com/facebookresearch/ijepa",
    checkpoint_format: CheckpointFormat::PyTorchStateDict,
    pretrained_on: "ImageNet-22K",
};

/// V-JEPA ViT-Large/16 pretrained on VideoMix2M.
pub const VJEPA_VIT_L16: PretrainedModel = PretrainedModel {
    name: "V-JEPA ViT-L/16",
    architecture: "ViT-Large/16",
    family: ModelFamily::VJepa,
    resolution: (224, 224),
    patch_size: (16, 16),
    embed_dim: 1024,
    num_layers: 24,
    num_heads: 16,
    mlp_dim: 4096,
    num_params: 304_000_000,
    huggingface_url: None,
    weights_url: Some("https://dl.fbaipublicfiles.com/jepa/vit.l.16-k400-300e.pth.tar"),
    source_repo: "https://github.com/facebookresearch/jepa",
    checkpoint_format: CheckpointFormat::PyTorchStateDict,
    pretrained_on: "VideoMix2M",
};

/// V-JEPA ViT-Huge/16 pretrained on VideoMix2M.
pub const VJEPA_VIT_H16: PretrainedModel = PretrainedModel {
    name: "V-JEPA ViT-H/16",
    architecture: "ViT-Huge/16",
    family: ModelFamily::VJepa,
    resolution: (224, 224),
    patch_size: (16, 16),
    embed_dim: 1280,
    num_layers: 32,
    num_heads: 16,
    mlp_dim: 5120,
    num_params: 632_000_000,
    huggingface_url: None,
    weights_url: Some("https://dl.fbaipublicfiles.com/jepa/vit.h.16-k400-300e.pth.tar"),
    source_repo: "https://github.com/facebookresearch/jepa",
    checkpoint_format: CheckpointFormat::PyTorchStateDict,
    pretrained_on: "VideoMix2M",
};

/// List all known pretrained JEPA models.
pub fn list_models() -> Vec<&'static PretrainedModel> {
    vec![
        &IJEPA_VIT_H14_IN1K,
        &IJEPA_VIT_H16_448_IN1K,
        &IJEPA_VIT_H14_IN22K,
        &IJEPA_VIT_G16_IN22K,
        &VJEPA_VIT_L16,
        &VJEPA_VIT_H16,
    ]
}

/// List only I-JEPA models.
pub fn list_ijepa_models() -> Vec<&'static PretrainedModel> {
    list_models()
        .into_iter()
        .filter(|m| m.family == ModelFamily::IJepa)
        .collect()
}

/// List only V-JEPA models.
pub fn list_vjepa_models() -> Vec<&'static PretrainedModel> {
    list_models()
        .into_iter()
        .filter(|m| m.family == ModelFamily::VJepa)
        .collect()
}

/// Look up a model by name (case-insensitive substring match).
pub fn find_model(query: &str) -> Option<&'static PretrainedModel> {
    let query_lower = query.to_lowercase();
    list_models()
        .into_iter()
        .find(|m| m.name.to_lowercase().contains(&query_lower))
}

/// Print a formatted table of all available models.
pub fn format_model_table() -> String {
    let mut table = String::new();
    table.push_str(
        "┌─────────────────────────────┬──────────────┬───────────┬────────┬──────────────┐\n",
    );
    table.push_str(
        "│ Model                       │ Architecture │ Params    │ Input  │ Dataset      │\n",
    );
    table.push_str(
        "├─────────────────────────────┼──────────────┼───────────┼────────┼──────────────┤\n",
    );

    for model in list_models() {
        table.push_str(&format!(
            "│ {:<27} │ {:<12} │ {:<9} │ {:>3}x{:<3}│ {:<12} │\n",
            model.name,
            model.architecture,
            model.param_count_human(),
            model.resolution.0,
            model.resolution.1,
            model.pretrained_on,
        ));
    }

    table.push_str(
        "└─────────────────────────────┴──────────────┴───────────┴────────┴──────────────┘\n",
    );
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_models_non_empty() {
        assert!(!list_models().is_empty());
    }

    #[test]
    fn test_ijepa_models_are_ijepa() {
        for model in list_ijepa_models() {
            assert_eq!(model.family, ModelFamily::IJepa);
        }
    }

    #[test]
    fn test_vjepa_models_are_vjepa() {
        for model in list_vjepa_models() {
            assert_eq!(model.family, ModelFamily::VJepa);
        }
    }

    #[test]
    fn test_param_count_human_millions() {
        assert_eq!(IJEPA_VIT_H14_IN1K.param_count_human(), "632M");
    }

    #[test]
    fn test_param_count_human_billions() {
        assert_eq!(IJEPA_VIT_G16_IN22K.param_count_human(), "1.0B");
    }

    #[test]
    fn test_num_patches_h14() {
        // 224 / 14 = 16, 16 * 16 = 256
        assert_eq!(IJEPA_VIT_H14_IN1K.num_patches(), 256);
    }

    #[test]
    fn test_num_patches_h16_448() {
        // 448 / 16 = 28, 28 * 28 = 784
        assert_eq!(IJEPA_VIT_H16_448_IN1K.num_patches(), 784);
    }

    #[test]
    fn test_num_patches_g16() {
        // 224 / 16 = 14, 14 * 14 = 196
        assert_eq!(IJEPA_VIT_G16_IN22K.num_patches(), 196);
    }

    #[test]
    fn test_find_model_by_name() {
        let model = find_model("vit-h/14").unwrap();
        assert_eq!(model.embed_dim, 1280);
        assert_eq!(model.patch_size, (14, 14));
    }

    #[test]
    fn test_find_model_case_insensitive() {
        let model = find_model("VIT-G/16").unwrap();
        assert_eq!(model.family, ModelFamily::IJepa);
    }

    #[test]
    fn test_find_model_not_found() {
        assert!(find_model("nonexistent-model-xyz").is_none());
    }

    #[test]
    fn test_all_models_have_source_repo() {
        for model in list_models() {
            assert!(!model.source_repo.is_empty());
        }
    }

    #[test]
    fn test_all_models_have_weights_url() {
        for model in list_models() {
            assert!(model.weights_url.is_some());
        }
    }

    #[test]
    fn test_model_family_display() {
        assert_eq!(format!("{}", ModelFamily::IJepa), "I-JEPA");
        assert_eq!(format!("{}", ModelFamily::VJepa), "V-JEPA");
        assert_eq!(format!("{}", ModelFamily::VJepa2), "V-JEPA 2");
    }

    #[test]
    fn test_checkpoint_format_display() {
        assert_eq!(
            format!("{}", CheckpointFormat::PyTorchStateDict),
            "PyTorch (.pth)"
        );
        assert_eq!(
            format!("{}", CheckpointFormat::SafeTensors),
            "SafeTensors (.safetensors)"
        );
    }

    #[test]
    fn test_format_model_table() {
        let table = format_model_table();
        assert!(table.contains("I-JEPA"));
        assert!(table.contains("V-JEPA"));
        assert!(table.contains("632M"));
    }

    #[test]
    fn test_model_dimensions_consistent() {
        for model in list_models() {
            assert!(model.embed_dim > 0);
            assert!(model.num_layers > 0);
            assert!(model.num_heads > 0);
            assert_eq!(
                model.embed_dim % model.num_heads,
                0,
                "{}: embed_dim must be divisible by num_heads",
                model.name
            );
            assert!(model.resolution.0 % model.patch_size.0 == 0);
            assert!(model.resolution.1 % model.patch_size.1 == 0);
        }
    }
}
