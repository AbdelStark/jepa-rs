pub mod cli;
pub mod commands;
pub mod demo;
mod fmt_utils;
mod tui;

use anyhow::Result;

pub use cli::{Cli, Command};

pub fn run_command(cmd: Command) -> Result<()> {
    match cmd {
        Command::Models(args) => commands::models::run(args),
        Command::Inspect(args) => commands::inspect::run(args),
        Command::Checkpoint(args) => commands::checkpoint::run(args),
        Command::Train(args) => commands::train::run(*args),
        Command::Encode(args) => commands::encode::run(args),
        Command::Tui => tui::run(),
    }
}

pub fn run_tui() -> Result<()> {
    tui::run()
}
