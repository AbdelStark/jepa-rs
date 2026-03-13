use anyhow::Result;

use jepa_compat::registry::{self, ModelFamily};

use crate::cli::{ModelFamilyFilter, ModelsArgs};
use crate::fmt_utils::format_params;

pub fn run(args: ModelsArgs) -> Result<()> {
    let all_models = registry::list_models();

    let models: Vec<_> = match &args.family {
        Some(ModelFamilyFilter::Ijepa) => all_models
            .into_iter()
            .filter(|m| matches!(m.family, ModelFamily::IJepa))
            .collect(),
        Some(ModelFamilyFilter::Vjepa) => all_models
            .into_iter()
            .filter(|m| matches!(m.family, ModelFamily::VJepa))
            .collect(),
        None => all_models,
    };

    if let Some(ref query) = args.name {
        match registry::find_model(query) {
            Some(m) => {
                println!("┌─────────────────────────────────────────────┐");
                println!("│  Model: {:<35} │", m.name);
                println!("├─────────────────────────────────────────────┤");
                println!("│  Family:       {:<28} │", format!("{:?}", m.family));
                println!("│  Architecture: {:<28} │", m.architecture);
                println!(
                    "│  Parameters:   {:<28} │",
                    format_params(m.num_params as usize)
                );
                println!(
                    "│  Format:       {:<28} │",
                    format!("{:?}", m.checkpoint_format)
                );
                println!("└─────────────────────────────────────────────┘");
            }
            None => {
                eprintln!("No model found matching '{query}'");
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    println!();
    println!("  ╔══════════════════════════════════════════════════════════════════╗");
    println!("  ║                    JEPA Pretrained Models                       ║");
    println!("  ╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "  ║ {:<18} {:<10} {:<16} {:<16} ║",
        "Name", "Family", "Architecture", "Params"
    );
    println!("  ╠══════════════════════════════════════════════════════════════════╣");

    for m in &models {
        println!(
            "  ║ {:<18} {:<10} {:<16} {:<16} ║",
            m.name,
            format!("{:?}", m.family),
            m.architecture,
            format_params(m.num_params as usize),
        );
    }

    println!("  ╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "  {} model(s) found. Use --name <query> for details.",
        models.len()
    );
    println!();

    Ok(())
}
