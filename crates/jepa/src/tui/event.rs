use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, KeyEvent};

pub enum Event {
    Tick,
    Key(KeyEvent),
    Mouse,
    Resize,
}

pub struct EventHandler {
    rx: mpsc::Receiver<Event>,
    // Keep handle alive
    _tx: mpsc::Sender<Event>,
}

impl EventHandler {
    pub fn new(tick_rate_ms: u64) -> Self {
        let tick_rate = Duration::from_millis(tick_rate_ms);
        let (tx, rx) = mpsc::channel();
        let _tx = tx.clone();

        thread::spawn(move || loop {
            if event::poll(tick_rate).unwrap_or(false) {
                match event::read() {
                    Ok(event::Event::Key(key)) => {
                        if tx.send(Event::Key(key)).is_err() {
                            break;
                        }
                    }
                    Ok(event::Event::Mouse(_)) => {
                        if tx.send(Event::Mouse).is_err() {
                            break;
                        }
                    }
                    Ok(event::Event::Resize(_, _)) => {
                        if tx.send(Event::Resize).is_err() {
                            break;
                        }
                    }
                    _ => {}
                }
            } else if tx.send(Event::Tick).is_err() {
                break;
            }
        });

        Self { rx, _tx }
    }

    pub fn next(&self) -> Result<Event> {
        Ok(self.rx.recv()?)
    }
}
