use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;

use chrono::Local;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::commands::train::{
    self, TrainReporter, TrainRunSummary, TrainSourceKind, TrainStepMetrics,
};
use crate::demo::{self, DemoId, PreparedDemoDataset};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Dashboard,
    Models,
    Training,
    Checkpoint,
    About,
}

impl Tab {
    pub const ALL: [Tab; 5] = [
        Tab::Dashboard,
        Tab::Models,
        Tab::Training,
        Tab::Checkpoint,
        Tab::About,
    ];

    pub fn title(&self) -> &'static str {
        match self {
            Tab::Dashboard => "Dashboard",
            Tab::Models => "Models",
            Tab::Training => "Training",
            Tab::Checkpoint => "Checkpoint",
            Tab::About => "About",
        }
    }

    pub fn shortcut(&self) -> &'static str {
        match self {
            Tab::Dashboard => "1",
            Tab::Models => "2",
            Tab::Training => "3",
            Tab::Checkpoint => "4",
            Tab::About => "5",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingState {
    Idle,
    Running,
    Complete,
    Failed,
}

#[derive(Debug, Clone)]
enum DemoEvent {
    Log(String),
    DatasetPrepared(PreparedDemoDataset),
    TrainStarted(TrainRunSummary),
    TrainStep(TrainStepMetrics),
    Completed(String),
    Failed(String),
}

pub struct TrainingStatus {
    pub state: TrainingState,
    pub selected_demo_index: usize,
    pub current_step: usize,
    pub total_steps: usize,
    pub losses: Vec<f64>,
    pub energies: Vec<f64>,
    pub learning_rates: Vec<f64>,
    pub ema_momentums: Vec<f64>,
    pub logs: Vec<String>,
    pub summary_lines: Vec<String>,
    pub phase_title: String,
    pub phase_detail: String,
    pub progress_ratio: f64,
    pub run_summary: Option<TrainRunSummary>,
    pub prepared_dataset: Option<PreparedDemoDataset>,
    pub last_error: Option<String>,
    progress_offset: f64,
}

impl TrainingStatus {
    pub fn new() -> Self {
        let mut status = Self {
            state: TrainingState::Idle,
            selected_demo_index: 0,
            current_step: 0,
            total_steps: 0,
            losses: Vec::new(),
            energies: Vec::new(),
            learning_rates: Vec::new(),
            ema_momentums: Vec::new(),
            logs: Vec::new(),
            summary_lines: Vec::new(),
            phase_title: String::new(),
            phase_detail: String::new(),
            progress_ratio: 0.0,
            run_summary: None,
            prepared_dataset: None,
            last_error: None,
            progress_offset: 0.0,
        };
        status.preview_selected_demo();
        status
    }

    pub fn selected_demo(&self) -> DemoId {
        DemoId::ALL[self.selected_demo_index]
    }

    fn can_change_selection(&self) -> bool {
        self.state != TrainingState::Running
    }

    fn select_previous_demo(&mut self) {
        if !self.can_change_selection() {
            return;
        }

        if self.selected_demo_index > 0 {
            self.selected_demo_index -= 1;
            self.preview_selected_demo();
        }
    }

    fn select_next_demo(&mut self) {
        if !self.can_change_selection() {
            return;
        }

        if self.selected_demo_index + 1 < DemoId::ALL.len() {
            self.selected_demo_index += 1;
            self.preview_selected_demo();
        }
    }

    fn preview_selected_demo(&mut self) {
        let demo = self.selected_demo();
        self.state = TrainingState::Idle;
        self.current_step = 0;
        self.total_steps = 0;
        self.losses.clear();
        self.energies.clear();
        self.learning_rates.clear();
        self.ema_momentums.clear();
        self.logs.clear();
        self.summary_lines = demo
            .process_notes()
            .iter()
            .map(|line| (*line).to_string())
            .collect();
        self.phase_title = "Guided demo ready".to_string();
        self.phase_detail = demo.subtitle().to_string();
        self.progress_ratio = 0.0;
        self.run_summary = None;
        self.prepared_dataset = None;
        self.last_error = None;
        self.progress_offset = 0.0;
    }

    fn start_run(&mut self) {
        let demo = self.selected_demo();
        self.state = TrainingState::Running;
        self.current_step = 0;
        self.total_steps = 0;
        self.losses.clear();
        self.energies.clear();
        self.learning_rates.clear();
        self.ema_momentums.clear();
        self.logs.clear();
        self.summary_lines.clear();
        self.phase_title = format!("Running {}", demo.title());
        self.phase_detail = demo.subtitle().to_string();
        self.progress_ratio = 0.0;
        self.run_summary = None;
        self.prepared_dataset = None;
        self.last_error = None;
        self.progress_offset = 0.0;
        self.push_log(format!("Starting `{}`.", demo.example_name()));
    }

    fn clear_output(&mut self) {
        if self.state == TrainingState::Running {
            return;
        }
        self.preview_selected_demo();
    }

    fn push_log(&mut self, message: impl Into<String>) {
        let timestamp = Local::now().format("%H:%M:%S");
        self.logs.push(format!("{timestamp}  {}", message.into()));
        if self.logs.len() > 14 {
            let excess = self.logs.len() - 14;
            self.logs.drain(0..excess);
        }
    }

    fn apply_event(&mut self, event: DemoEvent) {
        match event {
            DemoEvent::Log(message) => self.push_log(message),
            DemoEvent::DatasetPrepared(prepared) => {
                self.progress_offset = 0.18;
                self.progress_ratio = self.progress_offset;
                self.phase_title = "Dataset prepared".to_string();
                self.phase_detail = format!(
                    "{} images ready under {}",
                    prepared.files.len(),
                    prepared.root.display()
                );
                self.push_log(format!(
                    "Prepared {} demo image(s) under {}.",
                    prepared.files.len(),
                    prepared.root.display()
                ));
                self.prepared_dataset = Some(prepared);
            }
            DemoEvent::TrainStarted(summary) => {
                self.total_steps = summary.steps;
                self.phase_title = "Training loop active".to_string();
                self.phase_detail = format!(
                    "{:?} preset, batch {}, source {}",
                    summary.preset, summary.batch_size, summary.source_description
                );
                self.push_log(format!(
                    "Training started with {:?} over {}.",
                    summary.preset, summary.source_description
                ));
                self.run_summary = Some(summary);
            }
            DemoEvent::TrainStep(metrics) => {
                self.current_step = metrics.step + 1;
                self.total_steps = metrics.total_steps;
                self.losses.push(metrics.total_loss);
                self.energies.push(metrics.energy);
                self.learning_rates.push(metrics.learning_rate);

                let momentum = if let Some(summary) = &self.run_summary {
                    let total = metrics.total_steps.max(1) as f64;
                    summary.ema_momentum
                        + (1.0 - summary.ema_momentum)
                            * 0.5
                            * (1.0 + (std::f64::consts::PI * metrics.step as f64 / total).cos())
                } else {
                    0.0
                };
                self.ema_momentums.push(momentum);

                let progress = self.current_step as f64 / metrics.total_steps.max(1) as f64;
                self.progress_ratio =
                    self.progress_offset + (1.0 - self.progress_offset) * progress;
                self.phase_title = "Monitoring strict optimization".to_string();
                self.phase_detail = format!(
                    "step {}/{} • loss {:.4} • energy {:.4}",
                    self.current_step, self.total_steps, metrics.total_loss, metrics.energy
                );
                self.push_log(format!(
                    "step {:>2}/{:<2}  loss {:.4}  energy {:.4}  lr {:.2e}",
                    self.current_step,
                    self.total_steps,
                    metrics.total_loss,
                    metrics.energy,
                    metrics.learning_rate
                ));
            }
            DemoEvent::Completed(headline) => {
                self.state = TrainingState::Complete;
                self.progress_ratio = 1.0;
                self.phase_title = headline;
                self.phase_detail = "Review the summary and live log below.".to_string();
                self.summary_lines = self.build_completion_summary();
                self.push_log("Demo completed successfully.");
            }
            DemoEvent::Failed(message) => {
                self.state = TrainingState::Failed;
                self.phase_title = "Demo failed".to_string();
                self.phase_detail = message.clone();
                self.last_error = Some(message.clone());
                self.summary_lines = vec![
                    "The demo exited with an error.".to_string(),
                    "Check the live log for the last successful phase.".to_string(),
                    message.clone(),
                ];
                self.push_log(format!("Demo failed: {message}"));
            }
        }
    }

    fn build_completion_summary(&self) -> Vec<String> {
        let demo = self.selected_demo();
        let mut lines = Vec::new();

        match demo {
            DemoId::ImageFolderTraining => {
                lines.push("Strict image-folder training demo finished cleanly.".to_string());
                if let Some(prepared) = &self.prepared_dataset {
                    lines.push(format!(
                        "Prepared {} generated PNG files in {}.",
                        prepared.files.len(),
                        prepared.root.display()
                    ));
                }
                if let Some(summary) = &self.run_summary {
                    lines.push(format!(
                        "Ran {:?} for {} step(s) with {}.",
                        summary.preset, summary.steps, summary.source_description
                    ));
                }
                if let (Some(first), Some(last)) = (self.losses.first(), self.losses.last()) {
                    lines.push(format!("Loss moved from {:.4} to {:.4}.", first, last));
                }
                lines.push(
                    "This path exercised decode, RGB conversion, resize, crop, normalization, masking, optimizer updates, and EMA.".to_string(),
                );
            }
            DemoId::SyntheticTraining => {
                lines.push("Synthetic training demo finished cleanly.".to_string());
                if let Some(summary) = &self.run_summary {
                    lines.push(format!(
                        "Ran {:?} for {} step(s) with synthetic random tensors.",
                        summary.preset, summary.steps
                    ));
                }
                if let Some(last) = self.losses.last() {
                    lines.push(format!("Final total loss: {:.4}.", last));
                }
                lines.push(
                    "This is the quickest way to verify the strict optimizer, masking, predictor, and EMA path without dataset I/O.".to_string(),
                );
            }
            DemoId::PrepareImageFolder => {
                lines.push("Demo image dataset is ready.".to_string());
                if let Some(prepared) = &self.prepared_dataset {
                    lines.push(format!(
                        "Wrote {} PNG files under {}.",
                        prepared.files.len(),
                        prepared.root.display()
                    ));
                    lines.push(
                        "The nested directory layout can be passed directly to `jepa train --dataset-dir`."
                            .to_string(),
                    );
                }
                lines.push(
                    "This setup demo keeps the repository small while still giving the TUI and CLI a real recursive dataset to walk.".to_string(),
                );
            }
        }

        lines
    }
}

pub struct App {
    pub should_quit: bool,
    pub active_tab: Tab,
    pub tab_index: usize,
    pub tick_count: u64,
    pub model_list_index: usize,
    pub training: TrainingStatus,
    pub show_help: bool,
    pub sparkline_data: Vec<u64>,
    pub scroll_offset: usize,
    demo_rx: Option<Receiver<DemoEvent>>,
}

impl App {
    pub fn new() -> Self {
        Self {
            should_quit: false,
            active_tab: Tab::Dashboard,
            tab_index: 0,
            tick_count: 0,
            model_list_index: 0,
            training: TrainingStatus::new(),
            show_help: false,
            sparkline_data: vec![0; 60],
            scroll_offset: 0,
            demo_rx: None,
        }
    }

    pub fn on_tick(&mut self) {
        self.tick_count += 1;
        self.drain_demo_events();

        let val = if let Some(last_loss) = self.training.losses.last().copied() {
            ((last_loss.min(3.0) / 3.0) * 70.0).round() as u64 + 10
        } else {
            ((self.tick_count as f64 * 0.15).sin() * 30.0 + 35.0) as u64
        };
        self.sparkline_data.push(val);
        if self.sparkline_data.len() > 60 {
            self.sparkline_data.remove(0);
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => {
                if self.show_help {
                    self.show_help = false;
                } else {
                    self.should_quit = true;
                }
                return;
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
                return;
            }
            KeyCode::Char('?') => {
                self.show_help = !self.show_help;
                return;
            }
            _ => {}
        }

        if self.show_help {
            return;
        }

        match key.code {
            KeyCode::Char('1') => self.switch_tab(0),
            KeyCode::Char('2') => self.switch_tab(1),
            KeyCode::Char('3') => self.switch_tab(2),
            KeyCode::Char('4') => self.switch_tab(3),
            KeyCode::Char('5') => self.switch_tab(4),
            KeyCode::Tab => self.next_tab(),
            KeyCode::BackTab => self.prev_tab(),
            _ => {}
        }

        match self.active_tab {
            Tab::Models => self.on_key_models(key),
            Tab::Training => self.on_key_training(key),
            Tab::Checkpoint => self.on_key_checkpoint(key),
            _ => {}
        }
    }

    fn drain_demo_events(&mut self) {
        let Some(rx) = &self.demo_rx else {
            return;
        };

        let mut disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(event) => self.training.apply_event(event),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }

        if disconnected {
            self.demo_rx = None;
        }
    }

    fn switch_tab(&mut self, index: usize) {
        if index < Tab::ALL.len() {
            self.tab_index = index;
            self.active_tab = Tab::ALL[index];
            self.scroll_offset = 0;
        }
    }

    fn next_tab(&mut self) {
        self.tab_index = (self.tab_index + 1) % Tab::ALL.len();
        self.active_tab = Tab::ALL[self.tab_index];
        self.scroll_offset = 0;
    }

    fn prev_tab(&mut self) {
        self.tab_index = if self.tab_index == 0 {
            Tab::ALL.len() - 1
        } else {
            self.tab_index - 1
        };
        self.active_tab = Tab::ALL[self.tab_index];
        self.scroll_offset = 0;
    }

    fn on_key_models(&mut self, key: KeyEvent) {
        let model_count = jepa_compat::registry::list_models().len();
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.model_list_index > 0 {
                    self.model_list_index -= 1;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.model_list_index + 1 < model_count {
                    self.model_list_index += 1;
                }
            }
            _ => {}
        }
    }

    fn on_key_training(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.training.select_previous_demo(),
            KeyCode::Down | KeyCode::Char('j') => self.training.select_next_demo(),
            KeyCode::Enter | KeyCode::Char('s') => self.start_selected_demo(),
            KeyCode::Char('r') => self.restart_selected_demo(),
            KeyCode::Char('c') => self.training.clear_output(),
            _ => {}
        }
    }

    fn on_key_checkpoint(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.scroll_offset > 0 {
                    self.scroll_offset -= 1;
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.scroll_offset += 1;
            }
            _ => {}
        }
    }

    fn start_selected_demo(&mut self) {
        if self.training.state == TrainingState::Running {
            return;
        }

        let demo = self.training.selected_demo();
        let (tx, rx) = mpsc::channel();
        self.demo_rx = Some(rx);
        self.training.start_run();

        thread::spawn(move || {
            if let Err(err) = run_demo_worker(demo, tx.clone()) {
                let _ = tx.send(DemoEvent::Failed(err.to_string()));
            }
        });
    }

    fn restart_selected_demo(&mut self) {
        if self.training.state == TrainingState::Running {
            return;
        }
        self.start_selected_demo();
    }
}

fn run_demo_worker(demo: DemoId, tx: Sender<DemoEvent>) -> anyhow::Result<()> {
    let _ = tx.send(DemoEvent::Log(format!(
        "Launching demo `{}`.",
        demo.example_name()
    )));

    match demo {
        DemoId::PrepareImageFolder => {
            let prepared = demo::prepare_demo_image_folder()?;
            let _ = tx.send(DemoEvent::DatasetPrepared(prepared));
            let _ = tx.send(DemoEvent::Completed("Demo dataset prepared".to_string()));
        }
        DemoId::SyntheticTraining => {
            let args = demo::synthetic_demo_args();
            let mut reporter = ChannelTrainReporter::new(tx.clone());
            train::run_with_reporter(args, &mut reporter)?;
            let _ = tx.send(DemoEvent::Completed(
                "Synthetic training demo complete".to_string(),
            ));
        }
        DemoId::ImageFolderTraining => {
            let prepared = demo::prepare_demo_image_folder()?;
            let dataset_dir = prepared.root.clone();
            let _ = tx.send(DemoEvent::DatasetPrepared(prepared));
            let mut reporter = ChannelTrainReporter::new(tx.clone());
            train::run_with_reporter(demo::image_folder_demo_args(dataset_dir), &mut reporter)?;
            let _ = tx.send(DemoEvent::Completed(
                "Image-folder training demo complete".to_string(),
            ));
        }
    }

    Ok(())
}

struct ChannelTrainReporter {
    tx: Sender<DemoEvent>,
}

impl ChannelTrainReporter {
    fn new(tx: Sender<DemoEvent>) -> Self {
        Self { tx }
    }
}

impl TrainReporter for ChannelTrainReporter {
    fn on_run_started(&mut self, summary: &TrainRunSummary) {
        let _ = self.tx.send(DemoEvent::TrainStarted(summary.clone()));

        let source_label = match summary.source_kind {
            TrainSourceKind::Synthetic => "synthetic tensors",
            TrainSourceKind::Safetensors => "safetensors dataset",
            TrainSourceKind::ImageFolder => "image-folder dataset",
        };
        let _ = self.tx.send(DemoEvent::Log(format!(
            "Using {source_label}: {}",
            summary.source_description
        )));
    }

    fn on_step(&mut self, metrics: &TrainStepMetrics) {
        let _ = self.tx.send(DemoEvent::TrainStep(metrics.clone()));
    }

    fn on_run_complete(&mut self, _summary: &TrainRunSummary) {
        let _ = self.tx.send(DemoEvent::Log(
            "Training loop finished without errors.".to_string(),
        ));
    }
}
