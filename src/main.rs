extern crate chrono;
extern crate toml;
#[macro_use]
extern crate serde_derive;
extern crate winit;
extern crate vulkano_win;
#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate rayon;
extern crate cgmath;
extern crate obj;
extern crate specs;

#[macro_use]
mod helper;
mod logger;
mod config;
mod keycode;
mod graphics;
mod ecs;
mod objects;

use helper::*;
use logger::*;
use config::*;
use graphics::renderer::*;
use objects::*;
use ecs::{
    entities,
};

use winit::{
    Event, WindowEvent, DeviceEvent, KeyboardInput, MouseScrollDelta,
};
use vulkano::{
    instance::Version,
};
use rayon::{
    ThreadPoolBuilder,
};

#[cfg(debug_assertions)]
const DEBUG: bool = true;
#[cfg(not(debug_assertions))]
const DEBUG: bool = false;

const NAME: &'static str = "Game";
const VERSION: Version = Version {
    major: 0,
    minor: 1,
    patch: 0,
};

fn main() {
    // comment and uncomment to recompile shaders
    //assert!(true);

    init();
    let mut logger = Logger::new("log.txt");
    let pool = ThreadPoolBuilder::new().build()
        .unwrap_or_else(|err| logger.error("PoolCreate", err));
    let mut config = Config::new(&mut logger, "config.toml");
    let mut events_loop = winit::EventsLoop::new();
    let (mut renderer, _debug_callback) = Renderer::new(&mut logger, &events_loop);
    let objects = Objects::load(&mut logger, &renderer.queue);
    let mut ecs = ecs::init();

    for i in (0..360).step_by(10) {
        entities::create_obj(&mut ecs, Object::teapot, Vec3::new(0.0, 0.0, 0.0), 0.0, i as f32, 0.0);
    }
    
    if DEBUG {logger.warning("Debug", "This is a debug build, beware of any bugs or issues")}
    logger.info("Welcome", format!("{} {} - Made by Friz64", NAME, VERSION));

    let mut running = true;
    while running {
        // handle events
        events_loop.poll_events(|event| pool.install(|| match event {
            Event::WindowEvent {event, ..} => match event {
                WindowEvent::KeyboardInput {input: KeyboardInput {virtual_keycode, state, ..}, ..} => {
                    config.update_keys(virtual_keycode, state);
                },
                WindowEvent::MouseInput {button, state, ..} => {
                    config.update_mouse(button, state);
                },
                WindowEvent::CloseRequested => running = false,
                WindowEvent::Focused(foc) => renderer.focused = foc,
                _ => (),
            },
            Event::DeviceEvent {event, ..} => match event {
                DeviceEvent::MouseWheel {delta} if renderer.focused => if let MouseScrollDelta::LineDelta(_, y) = delta { 
                    config.update_scroll(y);
                },
                _ => (),
            },
            _ => (),
        }));
        
        // handle active inputs
        pool.install(|| {
            config.update_status();

            if config.controls.movement.forwards.down() {
                println!("forwards down");
            }
            if config.controls.movement.backwards.down() {
                println!("backwards down");
            }
            if config.controls.movement.left.down() {
                println!("left down");
            }
            if config.controls.movement.right.down() {
                println!("right down");
            }

            config.update_scroll(0.0);
        });

        if pool.install(|| {
            renderer.draw(&mut logger, &pool, &ecs, &objects)
        }) { continue; };
    };

    exit(&mut logger, 0);
}