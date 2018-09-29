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
extern crate simdnoise;

#[macro_use]
mod helper;
mod logger;
mod config;
mod graphics;
mod ecs;
mod objects;

use helper::*;
use logger::*;
use config::{
    configloader::{self, UpdateConfig},
};
use graphics::renderer::*;
use objects::*;
use ecs::{
    entities, components::*,
};

use std::{
    time::{Instant},
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
    assert!(true);

    init();
    let mut logger = Logger::new("log.txt");
    let pool = ThreadPoolBuilder::new().build()
        .unwrap_or_else(|err| logger.error("PoolCreate", err));
    let mut config = configloader::Config::new(&mut logger, "config.toml");
    let mut events_loop = winit::EventsLoop::new();
    let (mut renderer, _debug_callback) = Renderer::new(&mut logger, &events_loop);
    let objects = Objects::load(&mut logger, &renderer.queue);
    let mut ecs = ecs::init();
    let player = entities::create_player(&mut ecs, Vec3::new(10.0, 10.0, 125.0), 0.0, 0.0);

    /*let range: usize = 5;
    for x in 0..range {
        for y in 0..range {
            for z in 0..range {
                entities::create_obj(&mut ecs, Object::suzanne, Vec3::new(x as f32 * 5.0, y as f32 * 5.0, z as f32 * 5.0), 0.0, 0.0, 0.0);
            }
        }
    }*/

    let terrain = objects::gen_terrain(&mut logger, &renderer.queue);
    entities::create_obj(&mut ecs, terrain, Vec3::new(0.0, 0.0, 0.0), 0.0, 0.0, 0.0);
    
    if DEBUG {logger.warning("Debug", "This is a debug build, beware of any bugs or issues")}
    logger.info("Welcome", format!("{} {} - Made by Friz64", NAME, VERSION));
    
    let mut old_frame = Instant::now();
    let mut running = true;
    while running {
        // update deltatime
        let delta_time = pool.install(|| {
            let this_frame = Instant::now();
            let frametime = this_frame.duration_since(old_frame);
            old_frame = this_frame;

            frametime.as_secs() as f32 + frametime.subsec_nanos() as f32 / 1_000_000_000.0
        });

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
                DeviceEvent::MouseMotion {delta} if renderer.focused && renderer.cursor_grabbed => {
                    // update player pyr
                    let mut pyr_storage = ecs.write_storage::<PitchYawRoll>();
                    let mut player_pyr = pyr_storage.get_mut(player)
                        .unwrap_or_else(|| logger.error("PlayerController", "Invalid Player Entity; missing Pos Component"));
                    
                    let mouse_speed = config.controls.sensitivity.mouse_speed * 0.85;

                    player_pyr.0 -= delta.1 as f32 * mouse_speed * 0.005;
                    player_pyr.1 += delta.0 as f32 * mouse_speed * 0.005;

                    let max_pitch = std::f32::consts::FRAC_PI_2 - 0.0001;

                    if player_pyr.0 > max_pitch { // locks looking up
                        player_pyr.0 = max_pitch;
                    } else if player_pyr.0 < -max_pitch { // locks looking down
                        player_pyr.0 = -max_pitch;
                    }
                },
                _ => (),
            },
            _ => (),
        }));
        
        // updates down, hold, up and none
        pool.install(|| config.update_status());

        pool.install(|| {
            if config.controls.engine.grab_cursor.down() {
                renderer.cursor_grabbed = !renderer.cursor_grabbed;

                renderer.surface.window().grab_cursor(renderer.cursor_grabbed)
                    .unwrap_or_else(|err| logger.error("GrabCursor", err));

                renderer.surface.window().hide_cursor(renderer.cursor_grabbed);
            }
        });

        if pool.install(|| {
            renderer.draw(&mut logger, &pool, &delta_time, &ecs, &player, &objects, &config)
        }) { continue; };

        // sets scroll back to none
        pool.install(|| config.update_scroll(0.0));
    };

    exit(&mut logger, 0);
}