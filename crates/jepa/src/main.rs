use anyhow::Result;
use clap::Parser;
use jepa::{run_command, run_tui, Cli};

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(cmd) => run_command(cmd),
        None => run_tui(),
    }
}
