//! Shared formatting utilities for CLI commands and TUI.

/// Format a parameter count as a human-readable string (e.g. "86M", "1.3B").
pub(crate) fn format_params(count: usize) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1e9)
    } else if count >= 1_000_000 {
        format!("{:.0}M", count as f64 / 1e6)
    } else if count >= 1_000 {
        format!("{:.0}K", count as f64 / 1e3)
    } else {
        format!("{count}")
    }
}

/// Truncate a string to `max` characters, appending "..." if truncated.
pub(crate) fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max - 3])
    }
}
