use std::fs;
use std::io::ErrorKind;
use std::io::Write;
use nphysics3d::{
    world::World,
    math::Vector,
};
use log::warn;
use ansi_term::Color::Red;

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
                eprintln!("{}", Red.paint(format!("Can't create {} Folder", ::NAME)));
                shutdown!(1)
            },
        });
}

pub fn create_physics() -> World<f32> {
    let mut world = World::new();

    world.set_gravity(Vector::y() * -9.81);

    world
}