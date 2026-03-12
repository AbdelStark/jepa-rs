//! Browse the registry of pretrained Facebook Research JEPA models.
//!
//! Lists all known pretrained models with their architecture details,
//! download URLs, and parameter counts.
//!
//! ## Run
//!
//! ```bash
//! cargo run -p jepa-compat --example model_registry
//! ```

use jepa_compat::registry;

fn main() {
    println!("=== Pretrained JEPA Model Registry ===\n");

    // Print formatted table
    print!("{}", registry::format_model_table());
    println!();

    // Show detailed info for each model
    println!("--- Detailed Model Information ---\n");
    for model in registry::list_models() {
        println!("{}", model.name);
        println!("  Family:       {}", model.family);
        println!("  Architecture: {}", model.architecture);
        println!("  Parameters:   {}", model.param_count_human());
        println!(
            "  Resolution:   {}x{}",
            model.resolution.0, model.resolution.1
        );
        println!(
            "  Patch size:   {}x{}",
            model.patch_size.0, model.patch_size.1
        );
        println!("  Patches:      {}", model.num_patches());
        println!("  Embed dim:    {}", model.embed_dim);
        println!("  Layers:       {}", model.num_layers);
        println!("  Heads:        {}", model.num_heads);
        println!("  MLP dim:      {}", model.mlp_dim);
        println!("  Dataset:      {}", model.pretrained_on);
        println!("  Format:       {}", model.checkpoint_format);
        println!("  Source:       {}", model.source_repo);
        if let Some(url) = model.huggingface_url {
            println!("  HuggingFace:  {}", url);
        }
        if let Some(url) = model.weights_url {
            println!("  Weights:      {}", url);
        }
        println!();
    }

    // Show how to use the search
    println!("--- Model Search ---\n");
    let queries = ["vit-h/14", "giant", "v-jepa"];
    for query in &queries {
        match registry::find_model(query) {
            Some(model) => println!("  find_model(\"{}\") -> {}", query, model.name),
            None => println!("  find_model(\"{}\") -> not found", query),
        }
    }
    println!();

    // Show I-JEPA vs V-JEPA breakdown
    println!("--- By Family ---\n");
    println!("  I-JEPA models: {}", registry::list_ijepa_models().len());
    println!("  V-JEPA models: {}", registry::list_vjepa_models().len());
}
