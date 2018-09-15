use chrono::{DateTime, Datelike, Local, Timelike};
use std::fs::File;
use std::fmt;
use std::io::Write;

pub enum LogType {
    INFO,
    WARNING,
    ERROR,
}

impl LogType {
    pub fn gen_msg<T>(self, prefix: &str, msg: T) -> String
        where T: fmt::Display
    {
        let name = match self {
            LogType::INFO =>       "INFO",
            LogType::WARNING => "WARNING",
            LogType::ERROR =>     "ERROR",
        };

        format!("[{}] - {}: [{}] {}", format_time(Local::now()), name, prefix, msg)
    }
}

pub struct Logger {
    pub file: Option<File>,
}

impl Logger {
    pub fn new(logpath: &str) -> Self {
        let path = format!("./{}/{}", ::NAME, logpath);
        let file = File::create(&path)
            .map_err(|err| {
                println!("{}", LogType::WARNING.gen_msg("LogFileCreate", err.to_string() + " - NOT WRITING LOG TO FILE"))
            }).ok();

        let mut logger = Logger {
            file
        };
        logger.info("Logger", "Logger initialized");
        logger
    }

    pub fn error<T>(&mut self, prefix: &str, msg: T) -> !
        where T: fmt::Display
    {
        let err_msg = LogType::ERROR.gen_msg(prefix, msg);
        println!("{}", err_msg);
        if let Some(file) = &mut self.file {
            file.write(format!("{}\n", err_msg).as_bytes())
                .map_err(|err| {
                    println!("{}", ::LogType::WARNING.gen_msg("LogFileWrite", err))
                }).ok();;
        }
        
        ::exit(self, 1)
    }

    pub fn warning<T>(&mut self, prefix: &str, msg: T)
        where T: fmt::Display
    {
        let err_msg = LogType::WARNING.gen_msg(prefix, msg);
        println!("{}", err_msg);
        if let Some(file) = &mut self.file {
            file.write(format!("{}\n", err_msg).as_bytes())
                .map_err(|err| {
                    println!("{}", ::LogType::WARNING.gen_msg("LogFileWrite", err))
                }).ok();;
        }
    }

    pub fn info<T>(&mut self, prefix: &str, msg: T)
        where T: fmt::Display
    {
        let err_msg = LogType::INFO.gen_msg(prefix, msg);
        println!("{}", err_msg);
        if let Some(file) = &mut self.file {
            file.write(format!("{}\n", err_msg).as_bytes())
                .map_err(|err| {
                    println!("{}", ::LogType::WARNING.gen_msg("LogFileWrite", err))
                }).ok();;
        }
    }
}

fn format_time(time: DateTime<Local>) -> String {
    format!("{:02}.{:02}.{:04} {:02}:{:02}:{:02}",
        time.day(),
        time.month(),
        time.year(),
        time.hour(),
        time.minute(),
        time.second(),
    )
}