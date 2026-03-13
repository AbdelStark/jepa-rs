use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;

use chrono::Local;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::commands::encode::{
    self, InferenceDemoReporter, InferenceDemoRunSummary, InferencePhaseUpdate,
    InferenceSampleMetrics,
};
use crate::commands::train::{
    self, TrainReporter, TrainRunSummary, TrainSourceKind, TrainStepMetrics,
};
use crate::demo::{self, DemoId, InferenceDemoId, PreparedDemoDataset};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Dashboard,
    Models,
    Training,
    Inference,
    Checkpoint,
    About,
}

impl Tab {
    pub const ALL: [Tab; 6] = [
        Tab::Dashboard,
        Tab::Models,
        Tab::Training,
        Tab::Inference,
        Tab::Checkpoint,
        Tab::About,
    ];

    pub fn title(&self) -> &'static str {
        match self {
            Tab::Dashboard => "Dashboard",
            Tab::Models => "Models",
            Tab::Training => "Training",
            Tab::Inference => "Inference",
            Tab::Checkpoint => "Checkpoint",
            Tab::About => "About",
        }
    }

    pub fn shortcut(&self) -> &'static str {
        match self {
            Tab::Dashboard => "1",
            Tab::Models => "2",
            Tab::Training => "3",
            Tab::Inference => "4",
            Tab::Checkpoint => "5",
            Tab::About => "6",
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceState {
    Idle,
    Running,
    Complete,
    Failed,
}

#[derive(Debug, Clone)]
enum InferenceEvent {
    RunStarted(InferenceDemoRunSummary),
    Phase(InferencePhaseUpdate),
    Sample(InferenceSampleMetrics),
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

pub struct InferenceStatus {
    pub state: InferenceState,
    pub selected_demo_index: usize,
    pub current_sample: usize,
    pub total_samples: usize,
    pub inference_times: Vec<f64>,
    pub embedding_means: Vec<f64>,
    pub embedding_stds: Vec<f64>,
    pub mean_token_norms: Vec<f64>,
    pub logs: Vec<String>,
    pub summary_lines: Vec<String>,
    pub phase_title: String,
    pub phase_detail: String,
    pub progress_ratio: f64,
    pub run_summary: Option<InferenceDemoRunSummary>,
    pub last_sample_preview: Option<String>,
    pub last_error: Option<String>,
    progress_offset: f64,
}

impl InferenceStatus {
    pub fn new() -> Self {
        let mut status = Self {
            state: InferenceState::Idle,
            selected_demo_index: 0,
            current_sample: 0,
            total_samples: 0,
            inference_times: Vec::new(),
            embedding_means: Vec::new(),
            embedding_stds: Vec::new(),
            mean_token_norms: Vec::new(),
            logs: Vec::new(),
            summary_lines: Vec::new(),
            phase_title: String::new(),
            phase_detail: String::new(),
            progress_ratio: 0.0,
            run_summary: None,
            last_sample_preview: None,
            last_error: None,
            progress_offset: 0.0,
        };
        status.preview_selected_demo();
        status
    }

    pub fn selected_demo(&self) -> InferenceDemoId {
        InferenceDemoId::ALL[self.selected_demo_index]
    }

    fn can_change_selection(&self) -> bool {
        self.state != InferenceState::Running
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

        if self.selected_demo_index + 1 < InferenceDemoId::ALL.len() {
            self.selected_demo_index += 1;
            self.preview_selected_demo();
        }
    }

    fn preview_selected_demo(&mut self) {
        let demo = self.selected_demo();
        self.state = InferenceState::Idle;
        self.current_sample = 0;
        self.total_samples = 0;
        self.inference_times.clear();
        self.embedding_means.clear();
        self.embedding_stds.clear();
        self.mean_token_norms.clear();
        self.logs.clear();
        self.summary_lines = demo
            .process_notes()
            .iter()
            .map(|line| (*line).to_string())
            .collect();
        self.phase_title = "Inference walkthrough ready".to_string();
        self.phase_detail = demo.subtitle().to_string();
        self.progress_ratio = 0.0;
        self.run_summary = None;
        self.last_sample_preview = None;
        self.last_error = None;
        self.progress_offset = 0.0;
    }

    fn start_run(&mut self) {
        let demo = self.selected_demo();
        self.state = InferenceState::Running;
        self.current_sample = 0;
        self.total_samples = 0;
        self.inference_times.clear();
        self.embedding_means.clear();
        self.embedding_stds.clear();
        self.mean_token_norms.clear();
        self.logs.clear();
        self.summary_lines.clear();
        self.phase_title = format!("Running {}", demo.title());
        self.phase_detail = demo.subtitle().to_string();
        self.progress_ratio = 0.0;
        self.run_summary = None;
        self.last_sample_preview = None;
        self.last_error = None;
        self.progress_offset = 0.0;
        self.push_log(format!(
            "Starting inference walkthrough for `{}`.",
            demo.title()
        ));
    }

    fn clear_output(&mut self) {
        if self.state == InferenceState::Running {
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

    fn apply_event(&mut self, event: InferenceEvent) {
        match event {
            InferenceEvent::RunStarted(summary) => {
                self.total_samples = summary.num_samples;
                self.progress_offset = 0.18;
                self.progress_ratio = self.progress_offset;
                self.phase_title = "Inference plan loaded".to_string();
                let (patch_h, patch_w) = summary.patch_size;
                self.phase_detail = format!(
                    "{:?} • {}x{} patches • dim {} • {}",
                    summary.preset, patch_h, patch_w, summary.embed_dim, summary.model_description
                );
                self.push_log(format!(
                    "Prepared {:?} with {} patch tokens over {}.",
                    summary.preset, summary.num_patches, summary.input_description
                ));
                self.run_summary = Some(summary);
            }
            InferenceEvent::Phase(phase) => {
                self.phase_title = phase.title.clone();
                self.phase_detail = phase.detail.clone();
                self.progress_ratio = self.progress_ratio.max(self.progress_offset.max(0.28));
                self.push_log(format!("{}.", phase.detail));
            }
            InferenceEvent::Sample(metrics) => {
                self.current_sample = metrics.sample_index + 1;
                self.total_samples = metrics.total_samples;
                self.inference_times.push(metrics.inference_time_ms);
                self.embedding_means.push(metrics.embedding_mean);
                self.embedding_stds.push(metrics.embedding_std);
                self.mean_token_norms.push(metrics.mean_token_l2_norm);
                self.progress_ratio = self.progress_offset
                    + (1.0 - self.progress_offset)
                        * (self.current_sample as f64 / self.total_samples.max(1) as f64);
                self.phase_title = "Inspecting embedding output".to_string();
                self.phase_detail = format!(
                    "sample {}/{} • {} • {:.2} ms",
                    self.current_sample,
                    self.total_samples,
                    metrics.sample_label,
                    metrics.inference_time_ms
                );
                self.last_sample_preview = Some(format!(
                    "{} → [{}×{}×{}], mask={}, token[0]={}",
                    metrics.sample_label,
                    metrics.output_shape[0],
                    metrics.output_shape[1],
                    metrics.output_shape[2],
                    if metrics.mask_present {
                        "present"
                    } else {
                        "none"
                    },
                    format_first_token_preview(&metrics.first_token_preview)
                ));
                self.push_log(format!(
                    "sample {:>2}/{:<2}  {:<14}  {:.2} ms  mean {:+.4}  std {:.4}  norm {:.4}  mask {}",
                    self.current_sample,
                    self.total_samples,
                    metrics.sample_label,
                    metrics.inference_time_ms,
                    metrics.embedding_mean,
                    metrics.embedding_std,
                    metrics.mean_token_l2_norm,
                    if metrics.mask_present { "yes" } else { "no" }
                ));
            }
            InferenceEvent::Completed(headline) => {
                self.state = InferenceState::Complete;
                self.progress_ratio = 1.0;
                self.phase_title = headline;
                self.phase_detail =
                    "Review the samples, charts, and interpretation below.".to_string();
                self.summary_lines = self.build_completion_summary();
                self.push_log("Inference walkthrough completed successfully.");
            }
            InferenceEvent::Failed(message) => {
                self.state = InferenceState::Failed;
                self.phase_title = "Inference demo failed".to_string();
                self.phase_detail = message.clone();
                self.last_error = Some(message.clone());
                self.summary_lines = vec![
                    "The inference walkthrough exited with an error.".to_string(),
                    "Check the live log for the last completed phase.".to_string(),
                    message.clone(),
                ];
                self.push_log(format!("Inference walkthrough failed: {message}"));
            }
        }
    }

    fn build_completion_summary(&self) -> Vec<String> {
        let demo = self.selected_demo();
        let mut lines = Vec::new();

        lines.push(format!(
            "{} finished cleanly with {:?}.",
            demo.title(),
            demo.preset()
        ));
        if let Some(summary) = &self.run_summary {
            let (patch_h, patch_w) = summary.patch_size;
            lines.push(format!(
                "Encoded {} sample(s) at {}x{} over {} patch tokens with {}x{} patches and dim {}.",
                summary.num_samples,
                summary.image_size.0,
                summary.image_size.1,
                summary.num_patches,
                patch_h,
                patch_w,
                summary.embed_dim
            ));
        }
        if let Some(avg_latency) = average(&self.inference_times) {
            lines.push(format!("Average inference time: {:.2} ms.", avg_latency));
        }
        if let Some(last_norm) = self.mean_token_norms.last() {
            lines.push(format!("Latest mean token L2 norm: {:.4}.", last_norm));
        }
        if let Some(preview) = &self.last_sample_preview {
            lines.push(format!("Last sample preview: {preview}."));
        }
        match demo {
            InferenceDemoId::PatternVitSmall => lines.push(
                "This walkthrough proves the patch embedding and encoder stack are producing stable representation statistics on deterministic inputs.".to_string(),
            ),
            InferenceDemoId::PatternVitBase => lines.push(
                "The larger preset keeps the same 14x14 patch grid but usually shows higher runtime and a different embedding distribution because the encoder width is larger.".to_string(),
            ),
        }
        lines.push(
            "Because demo mode uses random-initialized weights, treat these outputs as pipeline telemetry rather than semantic quality.".to_string(),
        );

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
    pub inference: InferenceStatus,
    pub show_help: bool,
    pub sparkline_data: Vec<u64>,
    pub scroll_offset: usize,
    demo_rx: Option<Receiver<DemoEvent>>,
    inference_rx: Option<Receiver<InferenceEvent>>,
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
            inference: InferenceStatus::new(),
            show_help: false,
            sparkline_data: vec![0; 60],
            scroll_offset: 0,
            demo_rx: None,
            inference_rx: None,
        }
    }

    pub fn on_tick(&mut self) {
        self.tick_count += 1;
        self.drain_demo_events();
        self.drain_inference_events();

        let val = if let Some(last_loss) = self.training.losses.last().copied() {
            ((last_loss.min(3.0) / 3.0) * 70.0).round() as u64 + 10
        } else if let Some(last_latency) = self.inference.inference_times.last().copied() {
            ((last_latency.min(80.0) / 80.0) * 70.0).round() as u64 + 10
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
            KeyCode::Char('6') => self.switch_tab(5),
            KeyCode::Tab => self.next_tab(),
            KeyCode::BackTab => self.prev_tab(),
            _ => {}
        }

        match self.active_tab {
            Tab::Models => self.on_key_models(key),
            Tab::Training => self.on_key_training(key),
            Tab::Inference => self.on_key_inference(key),
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

    fn drain_inference_events(&mut self) {
        let Some(rx) = &self.inference_rx else {
            return;
        };

        let mut disconnected = false;
        loop {
            match rx.try_recv() {
                Ok(event) => self.inference.apply_event(event),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }

        if disconnected {
            self.inference_rx = None;
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

    fn on_key_inference(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.inference.select_previous_demo(),
            KeyCode::Down | KeyCode::Char('j') => self.inference.select_next_demo(),
            KeyCode::Enter | KeyCode::Char('s') => self.start_selected_inference_demo(),
            KeyCode::Char('r') => self.restart_selected_inference_demo(),
            KeyCode::Char('c') => self.inference.clear_output(),
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

    fn start_selected_inference_demo(&mut self) {
        if self.inference.state == InferenceState::Running {
            return;
        }

        let demo = self.inference.selected_demo();
        let (tx, rx) = mpsc::channel();
        self.inference_rx = Some(rx);
        self.inference.start_run();

        thread::spawn(move || {
            if let Err(err) = run_inference_worker(demo, tx.clone()) {
                let _ = tx.send(InferenceEvent::Failed(err.to_string()));
            }
        });
    }

    fn restart_selected_inference_demo(&mut self) {
        if self.inference.state == InferenceState::Running {
            return;
        }
        self.start_selected_inference_demo();
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

fn run_inference_worker(demo: InferenceDemoId, tx: Sender<InferenceEvent>) -> anyhow::Result<()> {
    let mut reporter = ChannelInferenceReporter::new(tx.clone());
    encode::run_inference_demo_with_reporter(demo, &mut reporter)?;
    let _ = tx.send(InferenceEvent::Completed(format!(
        "{} complete",
        demo.title()
    )));
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

struct ChannelInferenceReporter {
    tx: Sender<InferenceEvent>,
}

impl ChannelInferenceReporter {
    fn new(tx: Sender<InferenceEvent>) -> Self {
        Self { tx }
    }
}

impl InferenceDemoReporter for ChannelInferenceReporter {
    fn on_run_started(&mut self, summary: &InferenceDemoRunSummary) {
        let _ = self.tx.send(InferenceEvent::RunStarted(summary.clone()));
    }

    fn on_phase(&mut self, phase: &InferencePhaseUpdate) {
        let _ = self.tx.send(InferenceEvent::Phase(phase.clone()));
    }

    fn on_sample(&mut self, metrics: &InferenceSampleMetrics) {
        let _ = self.tx.send(InferenceEvent::Sample(metrics.clone()));
    }

    fn on_run_complete(&mut self, _summary: &InferenceDemoRunSummary) {}
}

fn average(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

fn format_first_token_preview(values: &[f32]) -> String {
    let preview: Vec<String> = values.iter().map(|value| format!("{value:.3}")).collect();
    format!("[{}]", preview.join(", "))
}
