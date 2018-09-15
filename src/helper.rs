use logger::Logger;
use std::fs;
use std::io::ErrorKind;
use std::io::{Write};

macro_rules! shutdown {
    ($e:expr) => {{
        println!("Exiting...");
        ::std::process::exit($e);
    }}
}

pub fn init() {
    let root = format!("./{}/", ::NAME);
    
    fs::create_dir(&root)
        .unwrap_or_else(|err| match err.kind() {
            ErrorKind::AlreadyExists => (),
            _ => {
                println!("{}", ::LogType::ERROR.gen_msg("FoldersCreate", err));
                shutdown!(1)
            },
        });
}

pub fn exit(logger: &mut Logger, exit_code: i32) -> ! {
    if let Some(file) = &mut logger.file {
        file.write(b"Exiting...")
            .map_err(|err| {
                println!("{}", ::LogType::ERROR.gen_msg("LogFileWrite", err));
            }).ok();
    }
    shutdown!(exit_code)
}