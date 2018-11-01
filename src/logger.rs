use chrono::{DateTime, Local, Timelike};
use log::{Log, Metadata, Record, Level, LevelFilter};
use std::{sync::Mutex, fs::{File, OpenOptions}, io::Write};
use ansi_term::Color::*;

#[macro_export]
macro_rules! error_close {
    ($($arg:tt)*) => {{
        log::error!($($arg)*);
        log::info!("Exiting...");
        std::process::exit(1)
    }};
}

pub struct Logger {
    file: Option<Mutex<File>>,
}

impl Logger {
    pub fn init(path: &str) {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(format!("./{}/{}", ::NAME, path));

        if let Err(err) = &file {
            eprintln!("{}", Red.paint(format!("Not logging to {} - {}", path, err)));
        }

        let logger = Logger {
            file: file.ok().map(|option| Mutex::new(option)),
        };

        log::set_boxed_logger(Box::new(logger)).unwrap();
        log::set_max_level(LevelFilter::Trace);
    }
}

impl Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Trace
    }

    fn log(&self, record: &Record) {
        let blacklist = ["winit"];

        let target = record.target();
        for item in blacklist.iter() {
            if target.contains(item) {
                return;
            }
        }

        if self.enabled(record.metadata()) {
            let time = format_time(Local::now());

            if let Some(file) = &self.file {
                if let Ok(mut file) = file.lock() {
                    writeln!(file, "[{}] {}{} - {}: {}",
                        time,
                        record.target(),
                        record.line().map(|option| format!(":{}", option)).unwrap_or_default(),
                        record.level(),
                        record.args(),
                    );
                }
            }

            let level = match record.level() {
                Level::Error => Red.paint("ERROR"),
                Level::Warn => Yellow.paint("WARN"),
                Level::Info => Green.paint("INFO"),
                Level::Debug => Purple.paint("DEBUG"),
                Level::Trace => Cyan.paint("TRACE"),
            };

            // https://upload.wikimedia.org/wikipedia/commons/1/15/Xterm_256color_chart.svg
            println!("[{}] {}{} - {}: {}",
                Fixed(245).paint(time),
                record.target(),
                record.line().map(|option| format!(":{}", Fixed(172).paint(option.to_string()))).unwrap_or_default(),
                level,
                record.args(),
            );
        }
    }

    fn flush(&self) {
        if let Some(file) = &self.file {
            if let Ok(mut file) = file.lock() {
                file.flush().ok();
            }
        }
    }
}

fn format_time(time: DateTime<Local>) -> String {
    format!("{:02}:{:02}:{:02}.{:03}",
        time.hour(),
        time.minute(),
        time.second(),
        time.nanosecond() / 1_000_000, // reduce to 3 chars
    )
}
