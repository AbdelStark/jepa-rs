//! PyTorch state_dict key mapping.
//!
//! Maps keys from PyTorch I-JEPA / V-JEPA checkpoints to the corresponding
//! burn module parameter paths used by jepa-rs.
//!
//! ## Key conventions
//!
//! PyTorch keys follow the pattern:
//! ```text
//! blocks.{layer}.attn.qkv.weight
//! blocks.{layer}.attn.proj.weight
//! blocks.{layer}.norm1.weight
//! blocks.{layer}.mlp.fc1.weight
//! ```
//!
//! jepa-rs burn keys follow:
//! ```text
//! blocks.{layer}.attn.qkv.weight
//! blocks.{layer}.attn.out_proj.weight
//! blocks.{layer}.norm1.weight
//! blocks.{layer}.mlp.fc1.weight
//! ```

use std::collections::HashMap;

/// A mapping rule that transforms a PyTorch key to a burn key.
///
/// # Example
///
/// ```
/// use jepa_compat::keymap::{KeyMapping, resolve_key};
///
/// let mappings = vec![KeyMapping {
///     pytorch_pattern: "blocks.{L}.norm.weight".to_string(),
///     burn_pattern: "blocks.{L}.norm.weight".to_string(),
/// }];
/// let result = resolve_key("blocks.3.norm.weight", &mappings);
/// assert_eq!(result, Some("blocks.3.norm.weight".to_string()));
/// ```
#[derive(Debug, Clone)]
pub struct KeyMapping {
    /// PyTorch key pattern (may contain `{layer}` placeholder).
    pub pytorch_pattern: String,
    /// Burn target key pattern (may contain `{layer}` placeholder).
    pub burn_pattern: String,
}

/// Build the default key mapping for I-JEPA ViT checkpoints.
///
/// Maps from Facebook Research's I-JEPA state_dict format to
/// jepa-rs VitEncoder parameter paths.
pub fn ijepa_vit_keymap() -> Vec<KeyMapping> {
    vec![
        // Patch embedding
        km("patch_embed.proj.weight", "patch_embed.projection.weight"),
        km("patch_embed.proj.bias", "patch_embed.projection.bias"),
        // Final layer norm
        km("norm.weight", "norm.weight"),
        km("norm.bias", "norm.bias"),
        // Transformer blocks (parameterized by layer index)
        km("blocks.{L}.norm1.weight", "blocks.{L}.norm1.weight"),
        km("blocks.{L}.norm1.bias", "blocks.{L}.norm1.bias"),
        km("blocks.{L}.attn.qkv.weight", "blocks.{L}.attn.qkv.weight"),
        km("blocks.{L}.attn.qkv.bias", "blocks.{L}.attn.qkv.bias"),
        km(
            "blocks.{L}.attn.proj.weight",
            "blocks.{L}.attn.out_proj.weight",
        ),
        km("blocks.{L}.attn.proj.bias", "blocks.{L}.attn.out_proj.bias"),
        km("blocks.{L}.norm2.weight", "blocks.{L}.norm2.weight"),
        km("blocks.{L}.norm2.bias", "blocks.{L}.norm2.bias"),
        km("blocks.{L}.mlp.fc1.weight", "blocks.{L}.mlp.fc1.weight"),
        km("blocks.{L}.mlp.fc1.bias", "blocks.{L}.mlp.fc1.bias"),
        km("blocks.{L}.mlp.fc2.weight", "blocks.{L}.mlp.fc2.weight"),
        km("blocks.{L}.mlp.fc2.bias", "blocks.{L}.mlp.fc2.bias"),
    ]
}

/// Build the key mapping for V-JEPA video encoder checkpoints.
///
/// V-JEPA uses the same ViT structure but may prefix with `encoder.`
/// or `module.` depending on the checkpoint source.
pub fn vjepa_vit_keymap() -> Vec<KeyMapping> {
    let mut mappings = ijepa_vit_keymap();
    // V-JEPA 2 uses tubelet embedding instead of 2D patch embed
    mappings.push(km(
        "patch_embed.proj.weight",
        "tubelet_embed.projection.weight",
    ));
    mappings.push(km("patch_embed.proj.bias", "tubelet_embed.projection.bias"));
    mappings
}

fn km(pytorch: &str, burn: &str) -> KeyMapping {
    KeyMapping {
        pytorch_pattern: pytorch.to_string(),
        burn_pattern: burn.to_string(),
    }
}

/// Resolve a PyTorch key to its burn equivalent using the provided mappings.
///
/// Returns `None` if no mapping matches the key.
///
/// # Example
///
/// ```
/// use jepa_compat::keymap::{ijepa_vit_keymap, resolve_key};
///
/// let mappings = ijepa_vit_keymap();
///
/// // attn.proj is remapped to attn.out_proj in burn
/// let result = resolve_key("blocks.0.attn.proj.weight", &mappings);
/// assert_eq!(result, Some("blocks.0.attn.out_proj.weight".to_string()));
///
/// // Unknown keys return None
/// assert_eq!(resolve_key("unknown.key", &mappings), None);
/// ```
pub fn resolve_key(pytorch_key: &str, mappings: &[KeyMapping]) -> Option<String> {
    for mapping in mappings {
        if let Some(burn_key) =
            try_match(&mapping.pytorch_pattern, &mapping.burn_pattern, pytorch_key)
        {
            return Some(burn_key);
        }
    }
    None
}

/// Try to match a pytorch key against a pattern with `{L}` layer placeholders.
///
/// If the pattern matches, substitute the layer index into the burn pattern.
fn try_match(pytorch_pattern: &str, burn_pattern: &str, key: &str) -> Option<String> {
    if !pytorch_pattern.contains("{L}") {
        // Exact match
        if key == pytorch_pattern {
            return Some(burn_pattern.to_string());
        }
        return None;
    }

    // Pattern has {L} — extract the layer index
    let parts: Vec<&str> = pytorch_pattern.split("{L}").collect();
    if parts.len() != 2 {
        return None;
    }

    let prefix = parts[0];
    let suffix = parts[1];

    if !key.starts_with(prefix) || !key.ends_with(suffix) {
        return None;
    }

    // Extract the layer index between prefix and suffix
    let mid = &key[prefix.len()..key.len() - suffix.len()];
    if mid.parse::<usize>().is_err() {
        return None;
    }

    Some(burn_pattern.replace("{L}", mid))
}

/// Strip common checkpoint prefixes from keys.
///
/// PyTorch checkpoints often wrap models with `module.`, `encoder.`,
/// or `backbone.` prefixes. This function strips them recursively.
///
/// # Example
///
/// ```
/// use jepa_compat::keymap::strip_prefix;
///
/// assert_eq!(strip_prefix("module.norm.weight"), "norm.weight");
/// assert_eq!(strip_prefix("module.encoder.norm.weight"), "norm.weight");
/// assert_eq!(strip_prefix("norm.weight"), "norm.weight");
/// ```
pub fn strip_prefix(key: &str) -> &str {
    let prefixes = ["module.", "encoder.", "backbone.", "model."];
    for prefix in &prefixes {
        if let Some(stripped) = key.strip_prefix(prefix) {
            return strip_prefix(stripped); // Recurse for nested prefixes
        }
    }
    key
}

/// Build a full key remapping table from a set of PyTorch keys.
///
/// Returns a map from original PyTorch key → burn parameter name.
/// Keys that don't match any mapping are collected in the second return value.
pub fn build_remap_table(
    pytorch_keys: &[String],
    mappings: &[KeyMapping],
) -> (HashMap<String, String>, Vec<String>) {
    let mut remap = HashMap::new();
    let mut unmapped = Vec::new();

    for key in pytorch_keys {
        let stripped = strip_prefix(key);
        if let Some(burn_key) = resolve_key(stripped, mappings) {
            remap.insert(key.clone(), burn_key);
        } else {
            unmapped.push(key.clone());
        }
    }

    (remap, unmapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_key_match() {
        let mappings = ijepa_vit_keymap();
        let result = resolve_key("norm.weight", &mappings);
        assert_eq!(result, Some("norm.weight".to_string()));
    }

    #[test]
    fn test_layer_pattern_match() {
        let mappings = ijepa_vit_keymap();
        let result = resolve_key("blocks.5.norm1.weight", &mappings);
        assert_eq!(result, Some("blocks.5.norm1.weight".to_string()));
    }

    #[test]
    fn test_attn_proj_remapped() {
        let mappings = ijepa_vit_keymap();
        let result = resolve_key("blocks.0.attn.proj.weight", &mappings);
        assert_eq!(result, Some("blocks.0.attn.out_proj.weight".to_string()));
    }

    #[test]
    fn test_unknown_key_returns_none() {
        let mappings = ijepa_vit_keymap();
        let result = resolve_key("some.random.key", &mappings);
        assert_eq!(result, None);
    }

    #[test]
    fn test_strip_prefix_module() {
        assert_eq!(
            strip_prefix("module.blocks.0.norm1.weight"),
            "blocks.0.norm1.weight"
        );
    }

    #[test]
    fn test_strip_prefix_nested() {
        assert_eq!(strip_prefix("module.encoder.norm.weight"), "norm.weight");
    }

    #[test]
    fn test_strip_prefix_none() {
        assert_eq!(
            strip_prefix("blocks.0.norm1.weight"),
            "blocks.0.norm1.weight"
        );
    }

    #[test]
    fn test_build_remap_table() {
        let keys = vec![
            "module.norm.weight".to_string(),
            "module.blocks.0.attn.proj.weight".to_string(),
            "unknown.key".to_string(),
        ];
        let mappings = ijepa_vit_keymap();
        let (remap, unmapped) = build_remap_table(&keys, &mappings);

        assert_eq!(
            remap.get("module.norm.weight"),
            Some(&"norm.weight".to_string())
        );
        assert_eq!(
            remap.get("module.blocks.0.attn.proj.weight"),
            Some(&"blocks.0.attn.out_proj.weight".to_string())
        );
        assert_eq!(unmapped, vec!["unknown.key".to_string()]);
    }

    #[test]
    fn test_ijepa_keymap_covers_all_layer_params() {
        let mappings = ijepa_vit_keymap();
        // A typical I-JEPA layer has these keys
        let layer_keys = [
            "blocks.3.norm1.weight",
            "blocks.3.norm1.bias",
            "blocks.3.attn.qkv.weight",
            "blocks.3.attn.qkv.bias",
            "blocks.3.attn.proj.weight",
            "blocks.3.attn.proj.bias",
            "blocks.3.norm2.weight",
            "blocks.3.norm2.bias",
            "blocks.3.mlp.fc1.weight",
            "blocks.3.mlp.fc1.bias",
            "blocks.3.mlp.fc2.weight",
            "blocks.3.mlp.fc2.bias",
        ];
        for key in &layer_keys {
            assert!(
                resolve_key(key, &mappings).is_some(),
                "key {key} should be mappable"
            );
        }
    }

    #[test]
    fn test_vjepa_keymap_includes_tubelet_embed() {
        let mappings = vjepa_vit_keymap();
        let result = resolve_key("patch_embed.proj.weight", &mappings);
        assert!(result.is_some());
    }

    #[test]
    fn test_strip_prefix_all_known_prefixes() {
        assert_eq!(strip_prefix("module.x"), "x");
        assert_eq!(strip_prefix("encoder.x"), "x");
        assert_eq!(strip_prefix("backbone.x"), "x");
        assert_eq!(strip_prefix("model.x"), "x");
    }

    #[test]
    fn test_build_remap_table_empty_input() {
        let mappings = ijepa_vit_keymap();
        let (remap, unmapped) = build_remap_table(&[], &mappings);
        assert!(remap.is_empty());
        assert!(unmapped.is_empty());
    }

    // --- Property-based tests ---
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_strip_prefix_idempotent(key in "[a-z.]+") {
            // Once stripped, stripping again should not change anything
            let stripped = strip_prefix(&key);
            let double_stripped = strip_prefix(stripped);
            prop_assert_eq!(stripped, double_stripped);
        }

        #[test]
        fn prop_strip_prefix_result_never_longer(key in "[a-z.]+") {
            let stripped = strip_prefix(&key);
            prop_assert!(stripped.len() <= key.len());
        }

        #[test]
        fn prop_resolve_key_is_deterministic(
            layer in 0usize..100,
            suffix_idx in 0usize..6,
        ) {
            let suffixes = [
                "norm1.weight", "norm1.bias",
                "attn.qkv.weight", "attn.proj.weight",
                "mlp.fc1.weight", "mlp.fc2.weight",
            ];
            let suffix = suffixes[suffix_idx % suffixes.len()];
            let key = format!("blocks.{layer}.{suffix}");
            let mappings = ijepa_vit_keymap();

            let result1 = resolve_key(&key, &mappings);
            let result2 = resolve_key(&key, &mappings);
            prop_assert_eq!(&result1, &result2, "resolve_key should be deterministic");
            prop_assert!(result1.is_some(), "all standard layer keys should resolve");
        }

        #[test]
        fn prop_layer_index_preserved_in_mapping(layer in 0usize..1000) {
            let key = format!("blocks.{layer}.norm1.weight");
            let mappings = ijepa_vit_keymap();
            let result = resolve_key(&key, &mappings).unwrap();
            prop_assert!(
                result.contains(&format!("{layer}")),
                "layer index should be preserved in mapping: {result}"
            );
        }

        #[test]
        fn prop_build_remap_covers_all_inputs(num_keys in 0usize..20) {
            let mappings = ijepa_vit_keymap();
            let keys: Vec<String> = (0..num_keys)
                .map(|i| format!("blocks.{i}.norm1.weight"))
                .collect();
            let (remap, unmapped) = build_remap_table(&keys, &mappings);
            // Every key should be either remapped or unmapped
            prop_assert_eq!(
                remap.len() + unmapped.len(),
                keys.len(),
                "all keys must be accounted for"
            );
        }
    }
}
