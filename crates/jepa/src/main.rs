mod cli;
mod commands;
mod fmt_utils;
mod tui;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Command};

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(cmd) => run_command(cmd),
        None => tui::run(),
    }
}

fn run_command(cmd: Command) -> Result<()> {
    match cmd {
        Command::Models(args) => commands::models::run(args),
        Command::Inspect(args) => commands::inspect::run(args),
        Command::Checkpoint(args) => commands::checkpoint::run(args),
        Command::Train(args) => commands::train::run(args),
        Command::Encode(args) => commands::encode::run(args),
        Command::Tui => tui::run(),
    }
}
