use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

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
    Paused,
    Complete,
}

pub struct TrainingStatus {
    pub state: TrainingState,
    pub current_step: usize,
    pub total_steps: usize,
    pub losses: Vec<f64>,
    pub energies: Vec<f64>,
    pub learning_rates: Vec<f64>,
    pub ema_momentums: Vec<f64>,
}

impl TrainingStatus {
    pub fn new() -> Self {
        Self {
            state: TrainingState::Idle,
            current_step: 0,
            total_steps: 500,
            losses: Vec::new(),
            energies: Vec::new(),
            learning_rates: Vec::new(),
            ema_momentums: Vec::new(),
        }
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
        }
    }

    pub fn on_tick(&mut self) {
        self.tick_count += 1;

        // Animate sparkline with a smooth wave pattern
        let val = ((self.tick_count as f64 * 0.15).sin() * 30.0 + 35.0) as u64;
        self.sparkline_data.push(val);
        if self.sparkline_data.len() > 60 {
            self.sparkline_data.remove(0);
        }

        // Simulate training progress when running
        if self.training.state == TrainingState::Running {
            if self.training.current_step < self.training.total_steps {
                self.training.current_step += 1;
                let step = self.training.current_step as f64;
                let total = self.training.total_steps as f64;

                // Simulated decaying loss
                let loss = 2.0 * (-step / (total * 0.3)).exp() + 0.1 + (step * 0.5).sin() * 0.05;
                self.training.losses.push(loss);

                let energy = 1.5 * (-step / (total * 0.4)).exp() + 0.05;
                self.training.energies.push(energy);

                // Warmup then cosine decay
                let warmup_frac = 0.1;
                let warmup_steps = total * warmup_frac;
                let lr = if step < warmup_steps {
                    1e-3 * step / warmup_steps
                } else {
                    let progress = (step - warmup_steps) / (total - warmup_steps);
                    1e-3 * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
                };
                self.training.learning_rates.push(lr);

                // Cosine momentum
                let momentum = 0.996
                    + (1.0 - 0.996) * 0.5 * (1.0 + (std::f64::consts::PI * step / total).cos());
                self.training.ema_momentums.push(momentum);
            } else {
                self.training.state = TrainingState::Complete;
            }
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        // Global keybinds
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

        // Tab switching
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

        // Per-tab keybinds
        match self.active_tab {
            Tab::Models => self.on_key_models(key),
            Tab::Training => self.on_key_training(key),
            Tab::Checkpoint => self.on_key_checkpoint(key),
            _ => {}
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
            KeyCode::Char('s') | KeyCode::Enter => match self.training.state {
                TrainingState::Idle | TrainingState::Complete => {
                    self.training = TrainingStatus::new();
                    self.training.state = TrainingState::Running;
                }
                TrainingState::Running => {
                    self.training.state = TrainingState::Paused;
                }
                TrainingState::Paused => {
                    self.training.state = TrainingState::Running;
                }
            },
            KeyCode::Char('r') => {
                self.training = TrainingStatus::new();
            }
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
}
