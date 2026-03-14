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
        Some(ModelFamilyFilter::Cjepa) => all_models
            .into_iter()
            .filter(|m| matches!(m.family, ModelFamily::CJepa))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_lists_all_models() {
        let args = ModelsArgs {
            family: None,
            name: None,
        };
        let result = run(args);
        assert!(result.is_ok());
    }

    #[test]
    fn run_filters_ijepa() {
        let args = ModelsArgs {
            family: Some(ModelFamilyFilter::Ijepa),
            name: None,
        };
        let result = run(args);
        assert!(result.is_ok());
    }

    #[test]
    fn run_filters_vjepa() {
        let args = ModelsArgs {
            family: Some(ModelFamilyFilter::Vjepa),
            name: None,
        };
        let result = run(args);
        assert!(result.is_ok());
    }

    #[test]
    fn run_filters_cjepa() {
        let args = ModelsArgs {
            family: Some(ModelFamilyFilter::Cjepa),
            name: None,
        };
        let result = run(args);
        assert!(result.is_ok());
    }
}
