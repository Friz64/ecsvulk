use ansi_term::Color::Red;
use nphysics3d::{math::Vector, world::World};
use std::fs;
use std::io::ErrorKind;

macro_rules! shutdown {
    ($e:expr) => {{
        println!("Exiting...");
        ::std::process::exit($e);
    }};
}

pub fn init() {
    let root = format!("./{}/", ::NAME);

    fs::create_dir(&root).unwrap_or_else(|err| match err.kind() {
        ErrorKind::AlreadyExists => (),
        _ => {
            eprintln!("{}", Red.paint(format!("Can't create {} Folder", ::NAME)));
            shutdown!(1)
        }
    });
}

pub fn log_system_info(device_name: String) {
    ::info!("System Information:");
    ::info!("- Graphics Device: {}", device_name);

    if let Ok(val) = sys_info::cpu_num() {
        ::info!("- CPU Threads: {}", val);
    };

    if let Ok(val) = sys_info::mem_info() {
        ::info!("- System RAM: {} GB", val.total as f32 / 1_000_000.0);
    };
}

pub fn create_physics() -> World<f32> {
    let mut world = World::new();

    world.set_gravity(Vector::y() * -9.81);

    world
}
