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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_params_billions() {
        assert_eq!(format_params(1_500_000_000), "1.5B");
        assert_eq!(format_params(1_000_000_000), "1.0B");
    }

    #[test]
    fn format_params_millions() {
        assert_eq!(format_params(86_000_000), "86M");
        assert_eq!(format_params(1_200_000), "1M");
    }

    #[test]
    fn format_params_thousands() {
        assert_eq!(format_params(50_000), "50K");
        assert_eq!(format_params(1_000), "1K");
    }

    #[test]
    fn format_params_small() {
        assert_eq!(format_params(999), "999");
        assert_eq!(format_params(0), "0");
        assert_eq!(format_params(1), "1");
    }

    #[test]
    fn truncate_short_string_unchanged() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("abc", 3), "abc");
    }

    #[test]
    fn truncate_long_string() {
        assert_eq!(truncate("hello world", 8), "hello...");
        assert_eq!(truncate("abcdefghij", 6), "abc...");
    }

    #[test]
    fn truncate_exact_length() {
        assert_eq!(truncate("abcde", 5), "abcde");
    }
}
