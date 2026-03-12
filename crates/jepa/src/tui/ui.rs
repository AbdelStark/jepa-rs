use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, BorderType, Borders, Chart, Clear, Dataset, Gauge, GraphType, List, ListItem,
    Padding, Paragraph, Sparkline, Tabs, Wrap,
};
use ratatui::Frame;

use super::app::{App, Tab, TrainingState};

// Color palette — Catppuccin Mocha
#[allow(dead_code)]
const LAVENDER: Color = Color::Rgb(180, 190, 254);
const MAUVE: Color = Color::Rgb(203, 166, 247);
const PINK: Color = Color::Rgb(245, 194, 231);
const PEACH: Color = Color::Rgb(250, 179, 135);
const YELLOW: Color = Color::Rgb(249, 226, 175);
const GREEN: Color = Color::Rgb(166, 227, 161);
const TEAL: Color = Color::Rgb(148, 226, 213);
const SAPPHIRE: Color = Color::Rgb(116, 199, 236);
const BLUE: Color = Color::Rgb(137, 180, 250);
const SURFACE0: Color = Color::Rgb(49, 50, 68);
const SURFACE1: Color = Color::Rgb(69, 71, 90);
const BASE: Color = Color::Rgb(30, 30, 46);
const TEXT: Color = Color::Rgb(205, 214, 244);
const SUBTEXT: Color = Color::Rgb(166, 173, 200);
const OVERLAY: Color = Color::Rgb(108, 112, 134);

pub fn draw(f: &mut Frame, app: &App) {
    let size = f.area();

    // Background
    f.render_widget(Block::default().style(Style::default().bg(BASE)), size);

    // Main layout: header, tabs, content, footer
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5), // Header/banner
            Constraint::Length(3), // Tabs
            Constraint::Min(10),   // Content
            Constraint::Length(1), // Footer
        ])
        .split(size);

    draw_header(f, main_chunks[0]);
    draw_tabs(f, main_chunks[1], app);
    draw_content(f, main_chunks[2], app);
    draw_footer(f, main_chunks[3], app);

    // Help overlay
    if app.show_help {
        draw_help_overlay(f, size);
    }
}

fn draw_header(f: &mut Frame, area: Rect) {
    let banner = vec![
        Line::from(vec![
            Span::styled(
                "       ░▀▀█ █▀▀ █▀█ █▀█   ",
                Style::default().fg(MAUVE).bold(),
            ),
            Span::styled("─── rs", Style::default().fg(LAVENDER)),
        ]),
        Line::from(vec![
            Span::styled(
                "       ░░█  █▀▀ █▀▀ █▀█   ",
                Style::default().fg(MAUVE).bold(),
            ),
            Span::styled(
                "Joint-Embedding Predictive Architecture",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(vec![
            Span::styled(
                "       █▄█  ▀▀▀ ▀   ▀ ▀   ",
                Style::default().fg(MAUVE).bold(),
            ),
            Span::styled("Toolkit for Rust", Style::default().fg(OVERLAY)),
        ]),
    ];

    let header = Paragraph::new(banner)
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_type(BorderType::Double)
                .border_style(Style::default().fg(SURFACE1))
                .padding(Padding::new(0, 0, 1, 0)),
        )
        .style(Style::default().bg(BASE));

    f.render_widget(header, area);
}

fn draw_tabs(f: &mut Frame, area: Rect, app: &App) {
    let titles: Vec<Line> = Tab::ALL
        .iter()
        .map(|t| {
            Line::from(vec![
                Span::styled(
                    format!(" {} ", t.shortcut()),
                    Style::default().fg(OVERLAY).add_modifier(Modifier::DIM),
                ),
                Span::raw(t.title()),
                Span::raw(" "),
            ])
        })
        .collect();

    let tabs = Tabs::new(titles)
        .select(app.tab_index)
        .style(Style::default().fg(SUBTEXT).bg(BASE))
        .highlight_style(
            Style::default()
                .fg(MAUVE)
                .bg(SURFACE0)
                .add_modifier(Modifier::BOLD),
        )
        .divider(Span::styled(" │ ", Style::default().fg(SURFACE1)));

    f.render_widget(tabs, area);
}

fn draw_content(f: &mut Frame, area: Rect, app: &App) {
    match app.active_tab {
        Tab::Dashboard => draw_dashboard(f, area, app),
        Tab::Models => draw_models(f, area, app),
        Tab::Training => draw_training(f, area, app),
        Tab::Checkpoint => draw_checkpoint(f, area, app),
        Tab::About => draw_about(f, area, app),
    }
}

fn draw_footer(f: &mut Frame, area: Rect, app: &App) {
    let keys = match app.active_tab {
        Tab::Dashboard => "Tab:switch  ?:help  q:quit",
        Tab::Models => "↑↓/jk:navigate  Tab:switch  ?:help  q:quit",
        Tab::Training => "s:start/pause  r:reset  Tab:switch  ?:help  q:quit",
        Tab::Checkpoint => "↑↓/jk:scroll  Tab:switch  ?:help  q:quit",
        Tab::About => "Tab:switch  ?:help  q:quit",
    };

    let footer = Paragraph::new(Line::from(vec![
        Span::styled(" JEPA ", Style::default().fg(BASE).bg(MAUVE).bold()),
        Span::styled(format!("  {keys}  "), Style::default().fg(SUBTEXT).bg(BASE)),
    ]));

    f.render_widget(footer, area);
}

// ── Dashboard ────────────────────────────────────────────────────────

fn draw_dashboard(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8), // Status cards
            Constraint::Min(8),    // Activity sparkline + info
        ])
        .margin(1)
        .split(area);

    // Status cards row
    let card_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(chunks[0]);

    draw_status_card(
        f,
        card_chunks[0],
        "Crates",
        "5",
        BLUE,
        "core vision world train compat",
    );
    draw_status_card(
        f,
        card_chunks[1],
        "Models",
        &format!("{}", jepa_compat::registry::list_models().len()),
        GREEN,
        "I-JEPA & V-JEPA",
    );
    draw_status_card(
        f,
        card_chunks[2],
        "Training",
        match app.training.state {
            TrainingState::Idle => "Idle",
            TrainingState::Running => "Running",
            TrainingState::Paused => "Paused",
            TrainingState::Complete => "Done",
        },
        match app.training.state {
            TrainingState::Idle => SUBTEXT,
            TrainingState::Running => GREEN,
            TrainingState::Paused => YELLOW,
            TrainingState::Complete => TEAL,
        },
        "Press 3 to view",
    );
    draw_status_card(
        f,
        card_chunks[3],
        "Backends",
        "NdArray",
        PEACH,
        "CPU • WGPU ready",
    );

    // Bottom section: sparkline + info
    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(chunks[1]);

    // Activity sparkline
    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .title(Line::from(vec![
                    Span::styled(" ◆ ", Style::default().fg(MAUVE)),
                    Span::styled("Activity", Style::default().fg(TEXT)),
                ]))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(SURFACE1))
                .padding(Padding::new(1, 1, 1, 1)),
        )
        .data(&app.sparkline_data)
        .style(Style::default().fg(LAVENDER).bg(BASE));

    f.render_widget(sparkline, bottom[0]);

    // Architecture overview
    let arch_lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  ┌─", Style::default().fg(SURFACE1)),
            Span::styled(" jepa-core ", Style::default().fg(BLUE)),
            Span::styled("─── traits, masking, energy", Style::default().fg(SUBTEXT)),
        ]),
        Line::from(vec![
            Span::styled("  ├─", Style::default().fg(SURFACE1)),
            Span::styled(" jepa-vision", Style::default().fg(GREEN)),
            Span::styled(" ── ViT, I-JEPA, V-JEPA", Style::default().fg(SUBTEXT)),
        ]),
        Line::from(vec![
            Span::styled("  ├─", Style::default().fg(SURFACE1)),
            Span::styled(" jepa-world ", Style::default().fg(PEACH)),
            Span::styled(" ── planning, hierarchy", Style::default().fg(SUBTEXT)),
        ]),
        Line::from(vec![
            Span::styled("  ├─", Style::default().fg(SURFACE1)),
            Span::styled(" jepa-train ", Style::default().fg(YELLOW)),
            Span::styled(" ── training, schedules", Style::default().fg(SUBTEXT)),
        ]),
        Line::from(vec![
            Span::styled("  └─", Style::default().fg(SURFACE1)),
            Span::styled(" jepa-compat", Style::default().fg(PINK)),
            Span::styled(" ── safetensors, ONNX", Style::default().fg(SUBTEXT)),
        ]),
    ];

    let arch = Paragraph::new(arch_lines).block(
        Block::default()
            .title(Line::from(vec![
                Span::styled(" ◆ ", Style::default().fg(MAUVE)),
                Span::styled("Architecture", Style::default().fg(TEXT)),
            ]))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(SURFACE1)),
    );

    f.render_widget(arch, bottom[1]);
}

fn draw_status_card(f: &mut Frame, area: Rect, title: &str, value: &str, color: Color, sub: &str) {
    let card = Block::default()
        .title(Line::from(vec![
            Span::styled(" ◆ ", Style::default().fg(color)),
            Span::styled(title, Style::default().fg(TEXT)),
        ]))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(SURFACE1));

    let inner = card.inner(area);
    f.render_widget(card, area);

    let content = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("  {value}"),
            Style::default()
                .fg(color)
                .bold()
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!("  {sub}"),
            Style::default().fg(OVERLAY),
        )),
    ]);

    f.render_widget(content, inner);
}

// ── Models ───────────────────────────────────────────────────────────

fn draw_models(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .margin(1)
        .split(area);

    let models = jepa_compat::registry::list_models();

    // Model list
    let items: Vec<ListItem> = models
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let style = if i == app.model_list_index {
                Style::default().fg(MAUVE).bg(SURFACE0).bold()
            } else {
                Style::default().fg(TEXT)
            };
            let prefix = if i == app.model_list_index {
                " ▸ "
            } else {
                "   "
            };
            ListItem::new(Line::from(vec![
                Span::styled(prefix, style),
                Span::styled(m.name, style),
            ]))
        })
        .collect();

    let model_list = List::new(items).block(
        Block::default()
            .title(Line::from(vec![
                Span::styled(" ◆ ", Style::default().fg(MAUVE)),
                Span::styled("Model Registry", Style::default().fg(TEXT)),
            ]))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(SURFACE1))
            .padding(Padding::new(0, 0, 1, 0)),
    );

    f.render_widget(model_list, chunks[0]);

    // Model details
    if let Some(selected) = models.get(app.model_list_index) {
        let param_str = format_params(selected.num_params as usize);
        let details = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  Name          ", Style::default().fg(OVERLAY)),
                Span::styled(selected.name, Style::default().fg(TEXT).bold()),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Family        ", Style::default().fg(OVERLAY)),
                Span::styled(format!("{:?}", selected.family), Style::default().fg(BLUE)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Architecture  ", Style::default().fg(OVERLAY)),
                Span::styled(selected.architecture, Style::default().fg(GREEN)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Parameters    ", Style::default().fg(OVERLAY)),
                Span::styled(&param_str, Style::default().fg(PEACH).bold()),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Format        ", Style::default().fg(OVERLAY)),
                Span::styled(
                    format!("{:?}", selected.checkpoint_format),
                    Style::default().fg(TEAL),
                ),
            ]),
            Line::from(""),
            Line::from(""),
            // Parameter bar chart (visual)
            Line::from(vec![
                Span::styled("  Scale  ", Style::default().fg(OVERLAY)),
                Span::styled(
                    param_bar(selected.num_params as usize),
                    Style::default().fg(MAUVE),
                ),
            ]),
        ];

        let detail_panel = Paragraph::new(details).block(
            Block::default()
                .title(Line::from(vec![
                    Span::styled(" ◆ ", Style::default().fg(MAUVE)),
                    Span::styled("Details", Style::default().fg(TEXT)),
                ]))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(SURFACE1)),
        );

        f.render_widget(detail_panel, chunks[1]);
    }
}

fn param_bar(count: usize) -> String {
    // Scale: max ~ 1B params
    let max = 1_000_000_000_f64;
    let ratio = (count as f64 / max).min(1.0);
    let bar_width = 30;
    let filled = (ratio * bar_width as f64) as usize;
    let empty = bar_width - filled;
    format!("{}{}", "█".repeat(filled), "░".repeat(empty),)
}

// ── Training ─────────────────────────────────────────────────────────

fn draw_training(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5), // Status bar
            Constraint::Min(10),   // Charts
            Constraint::Length(6), // Config
        ])
        .margin(1)
        .split(area);

    // Status/progress
    let progress = if app.training.total_steps > 0 {
        app.training.current_step as f64 / app.training.total_steps as f64
    } else {
        0.0
    };

    let state_color = match app.training.state {
        TrainingState::Idle => SUBTEXT,
        TrainingState::Running => GREEN,
        TrainingState::Paused => YELLOW,
        TrainingState::Complete => TEAL,
    };

    let state_text = match app.training.state {
        TrainingState::Idle => "IDLE — press s to start",
        TrainingState::Running => "TRAINING",
        TrainingState::Paused => "PAUSED — press s to resume",
        TrainingState::Complete => "COMPLETE",
    };

    let gauge = Gauge::default()
        .block(
            Block::default()
                .title(Line::from(vec![
                    Span::styled(" ◆ ", Style::default().fg(state_color)),
                    Span::styled(
                        format!(
                            "Training — {} — step {}/{}",
                            state_text, app.training.current_step, app.training.total_steps
                        ),
                        Style::default().fg(TEXT),
                    ),
                ]))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(SURFACE1)),
        )
        .gauge_style(Style::default().fg(MAUVE).bg(SURFACE0))
        .ratio(progress)
        .label(Span::styled(
            format!("{:.1}%", progress * 100.0),
            Style::default().fg(TEXT).bold(),
        ));

    f.render_widget(gauge, chunks[0]);

    // Charts area
    let chart_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    // Loss chart
    draw_metric_chart(f, chart_chunks[0], "Loss", &app.training.losses, PINK);

    // LR chart
    draw_metric_chart(
        f,
        chart_chunks[1],
        "Learning Rate",
        &app.training.learning_rates,
        SAPPHIRE,
    );

    // Config summary
    let last_loss = app.training.losses.last().copied().unwrap_or(0.0);
    let last_lr = app.training.learning_rates.last().copied().unwrap_or(0.0);
    let last_momentum = app.training.ema_momentums.last().copied().unwrap_or(0.0);

    let config_lines = vec![Line::from(vec![
        Span::styled("  Loss: ", Style::default().fg(OVERLAY)),
        Span::styled(format!("{last_loss:.6}"), Style::default().fg(PINK)),
        Span::styled("    LR: ", Style::default().fg(OVERLAY)),
        Span::styled(format!("{last_lr:.2e}"), Style::default().fg(SAPPHIRE)),
        Span::styled("    EMA momentum: ", Style::default().fg(OVERLAY)),
        Span::styled(format!("{last_momentum:.4}"), Style::default().fg(PEACH)),
        Span::styled("    Energy: ", Style::default().fg(OVERLAY)),
        Span::styled("L2", Style::default().fg(GREEN)),
        Span::styled("    Reg: ", Style::default().fg(OVERLAY)),
        Span::styled("VICReg", Style::default().fg(YELLOW)),
    ])];

    let config = Paragraph::new(config_lines).block(
        Block::default()
            .title(Line::from(vec![
                Span::styled(" ◆ ", Style::default().fg(MAUVE)),
                Span::styled("Metrics", Style::default().fg(TEXT)),
            ]))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(SURFACE1))
            .padding(Padding::new(0, 0, 1, 0)),
    );

    f.render_widget(config, chunks[2]);
}

fn draw_metric_chart(f: &mut Frame, area: Rect, title: &str, data: &[f64], color: Color) {
    if data.is_empty() {
        let placeholder = Paragraph::new(vec![
            Line::from(""),
            Line::from(""),
            Line::from(Span::styled(
                "  No data yet — start training",
                Style::default().fg(OVERLAY),
            )),
        ])
        .block(
            Block::default()
                .title(Line::from(vec![
                    Span::styled(" ◆ ", Style::default().fg(color)),
                    Span::styled(title, Style::default().fg(TEXT)),
                ]))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(SURFACE1)),
        );
        f.render_widget(placeholder, area);
        return;
    }

    let max_points = 100;
    let step = if data.len() > max_points {
        data.len() / max_points
    } else {
        1
    };

    let points: Vec<(f64, f64)> = data
        .iter()
        .enumerate()
        .step_by(step)
        .map(|(i, &v)| (i as f64, v))
        .collect();

    let y_min = data.iter().cloned().fold(f64::INFINITY, f64::min) * 0.9;
    let y_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 1.1;
    let y_max = if (y_max - y_min).abs() < 1e-10 {
        y_min + 1.0
    } else {
        y_max
    };
    let x_max = data.len() as f64;

    let dataset = Dataset::default()
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(color))
        .data(&points);

    let chart = Chart::new(vec![dataset])
        .block(
            Block::default()
                .title(Line::from(vec![
                    Span::styled(" ◆ ", Style::default().fg(color)),
                    Span::styled(title, Style::default().fg(TEXT)),
                ]))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(SURFACE1)),
        )
        .x_axis(
            Axis::default()
                .style(Style::default().fg(OVERLAY))
                .bounds([0.0, x_max])
                .labels(vec![
                    Span::styled("0", Style::default().fg(SUBTEXT)),
                    Span::styled(format!("{}", data.len()), Style::default().fg(SUBTEXT)),
                ]),
        )
        .y_axis(
            Axis::default()
                .style(Style::default().fg(OVERLAY))
                .bounds([y_min, y_max])
                .labels(vec![
                    Span::styled(format!("{y_min:.3}"), Style::default().fg(SUBTEXT)),
                    Span::styled(format!("{y_max:.3}"), Style::default().fg(SUBTEXT)),
                ]),
        );

    f.render_widget(chart, area);
}

// ── Checkpoint ───────────────────────────────────────────────────────

fn draw_checkpoint(f: &mut Frame, area: Rect, _app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Info
            Constraint::Min(6),     // Formats
        ])
        .margin(1)
        .split(area);

    let info_lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  The ", Style::default().fg(SUBTEXT)),
            Span::styled("jepa checkpoint", Style::default().fg(MAUVE).bold()),
            Span::styled(
                " command loads and analyzes model checkpoints.",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "  Supported formats:",
            Style::default().fg(TEXT),
        )]),
        Line::from(""),
        Line::from(vec![
            Span::styled("    ◆ ", Style::default().fg(GREEN)),
            Span::styled("Safetensors ", Style::default().fg(GREEN).bold()),
            Span::styled(
                "(.safetensors) — native checkpoint format",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(vec![
            Span::styled("    ◆ ", Style::default().fg(YELLOW)),
            Span::styled("ONNX         ", Style::default().fg(YELLOW).bold()),
            Span::styled(
                "(.onnx) — metadata inspection + initializer extraction",
                Style::default().fg(SUBTEXT),
            ),
        ]),
    ];

    let info = Paragraph::new(info_lines).block(
        Block::default()
            .title(Line::from(vec![
                Span::styled(" ◆ ", Style::default().fg(MAUVE)),
                Span::styled("Checkpoint Inspector", Style::default().fg(TEXT)),
            ]))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(SURFACE1)),
    );

    f.render_widget(info, chunks[0]);

    // Key mappings info
    let keymap_lines = vec![
        Line::from(""),
        Line::from(vec![Span::styled(
            "  Available key mappings for checkpoint conversion:",
            Style::default().fg(TEXT),
        )]),
        Line::from(""),
        Line::from(vec![
            Span::styled("    ◆ ", Style::default().fg(BLUE)),
            Span::styled("ijepa  ", Style::default().fg(BLUE).bold()),
            Span::styled(
                "— PyTorch I-JEPA ViT → burn key remapping",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(vec![
            Span::styled("    ◆ ", Style::default().fg(PEACH)),
            Span::styled("vjepa  ", Style::default().fg(PEACH).bold()),
            Span::styled(
                "— PyTorch V-JEPA ViT → burn key remapping",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(vec![
            Span::styled("    ◆ ", Style::default().fg(OVERLAY)),
            Span::styled("none   ", Style::default().fg(OVERLAY).bold()),
            Span::styled(
                "— load keys as-is without remapping",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  CLI usage: ", Style::default().fg(OVERLAY)),
            Span::styled(
                "jepa checkpoint model.safetensors --keymap ijepa --verbose",
                Style::default().fg(LAVENDER),
            ),
        ]),
    ];

    let keymap_panel = Paragraph::new(keymap_lines).block(
        Block::default()
            .title(Line::from(vec![
                Span::styled(" ◆ ", Style::default().fg(MAUVE)),
                Span::styled("Key Mappings", Style::default().fg(TEXT)),
            ]))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(SURFACE1)),
    );

    f.render_widget(keymap_panel, chunks[1]);
}

// ── About ────────────────────────────────────────────────────────────

fn draw_about(f: &mut Frame, area: Rect, _app: &App) {
    let content = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  jepa-rs", Style::default().fg(MAUVE).bold()),
            Span::styled(
                " — Joint-Embedding Predictive Architecture in Rust",
                Style::default().fg(TEXT),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Built on ", Style::default().fg(SUBTEXT)),
            Span::styled("burn 0.20", Style::default().fg(PEACH).bold()),
            Span::styled(" tensor framework", Style::default().fg(SUBTEXT)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  ─────────────────────────────────────────",
            Style::default().fg(SURFACE1),
        )),
        Line::from(""),
        Line::from(vec![Span::styled(
            "  Components:",
            Style::default().fg(TEXT).bold(),
        )]),
        Line::from(""),
        Line::from(vec![
            Span::styled("   ◆ ", Style::default().fg(BLUE)),
            Span::styled("jepa-core    ", Style::default().fg(BLUE)),
            Span::styled(
                "Traits, masking strategies, energy functions,",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(vec![
            Span::styled("                  ", Style::default().fg(BLUE)),
            Span::styled(
                "EMA, collapse regularization (VICReg, Barlow Twins)",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("   ◆ ", Style::default().fg(GREEN)),
            Span::styled("jepa-vision  ", Style::default().fg(GREEN)),
            Span::styled(
                "Vision Transformers, I-JEPA, V-JEPA,",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(vec![
            Span::styled("                  ", Style::default().fg(GREEN)),
            Span::styled(
                "patch embedding, RoPE, transformer predictor",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("   ◆ ", Style::default().fg(PEACH)),
            Span::styled("jepa-world   ", Style::default().fg(PEACH)),
            Span::styled(
                "Action-conditioned prediction, planning,",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(vec![
            Span::styled("                  ", Style::default().fg(PEACH)),
            Span::styled(
                "hierarchical JEPA, short-term memory",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("   ◆ ", Style::default().fg(YELLOW)),
            Span::styled("jepa-train   ", Style::default().fg(YELLOW)),
            Span::styled(
                "Training orchestration, warmup-cosine LR,",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(vec![
            Span::styled("                  ", Style::default().fg(YELLOW)),
            Span::styled(
                "checkpoint metadata, step-oriented design",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("   ◆ ", Style::default().fg(PINK)),
            Span::styled("jepa-compat  ", Style::default().fg(PINK)),
            Span::styled(
                "Safetensors loading, ONNX metadata/runtime,",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(vec![
            Span::styled("                  ", Style::default().fg(PINK)),
            Span::styled(
                "pretrained model registry, key remapping",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  ─────────────────────────────────────────",
            Style::default().fg(SURFACE1),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Status: ", Style::default().fg(OVERLAY)),
            Span::styled("alpha", Style::default().fg(YELLOW).bold()),
            Span::styled(
                " — for experimentation and extension",
                Style::default().fg(SUBTEXT),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Repository: ", Style::default().fg(OVERLAY)),
            Span::styled(
                "github.com/AbdelStark/jepa-rs",
                Style::default().fg(LAVENDER),
            ),
        ]),
        Line::from(vec![
            Span::styled("  License: ", Style::default().fg(OVERLAY)),
            Span::styled("MIT", Style::default().fg(TEXT)),
        ]),
    ];

    let about = Paragraph::new(content)
        .block(
            Block::default()
                .title(Line::from(vec![
                    Span::styled(" ◆ ", Style::default().fg(MAUVE)),
                    Span::styled("About JEPA-RS", Style::default().fg(TEXT)),
                ]))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(SURFACE1)),
        )
        .wrap(Wrap { trim: false });

    f.render_widget(about, area);
}

// ── Help Overlay ─────────────────────────────────────────────────────

fn draw_help_overlay(f: &mut Frame, area: Rect) {
    let overlay_width = 50;
    let overlay_height = 20;
    let x = area.width.saturating_sub(overlay_width) / 2;
    let y = area.height.saturating_sub(overlay_height) / 2;
    let overlay_area = Rect::new(
        x,
        y,
        overlay_width.min(area.width),
        overlay_height.min(area.height),
    );

    f.render_widget(Clear, overlay_area);

    let help_text = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  Keyboard Shortcuts",
            Style::default().fg(MAUVE).bold(),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  1-5      ", Style::default().fg(LAVENDER).bold()),
            Span::styled("Switch tabs", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  Tab      ", Style::default().fg(LAVENDER).bold()),
            Span::styled("Next tab", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  Shift+Tab", Style::default().fg(LAVENDER).bold()),
            Span::styled("Previous tab", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  j/k ↑/↓  ", Style::default().fg(LAVENDER).bold()),
            Span::styled("Navigate lists", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  s/Enter  ", Style::default().fg(LAVENDER).bold()),
            Span::styled("Start/pause training", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  r        ", Style::default().fg(LAVENDER).bold()),
            Span::styled("Reset training", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  ?        ", Style::default().fg(LAVENDER).bold()),
            Span::styled("Toggle help", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled("  q/Esc    ", Style::default().fg(LAVENDER).bold()),
            Span::styled("Quit", Style::default().fg(TEXT)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Press ? or Esc to close",
            Style::default().fg(OVERLAY),
        )),
    ];

    let help = Paragraph::new(help_text).block(
        Block::default()
            .title(Line::from(vec![
                Span::styled(" ◆ ", Style::default().fg(MAUVE)),
                Span::styled("Help", Style::default().fg(TEXT)),
            ]))
            .borders(Borders::ALL)
            .border_type(BorderType::Double)
            .border_style(Style::default().fg(MAUVE))
            .style(Style::default().bg(BASE)),
    );

    f.render_widget(help, overlay_area);
}

fn format_params(count: usize) -> String {
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
