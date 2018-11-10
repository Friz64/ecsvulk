extern crate chrono;
extern crate toml;
#[macro_use]
extern crate serde_derive;
extern crate vulkano_win;
extern crate winit;
#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate ansi_term;
extern crate cgmath;
extern crate log;
extern crate nalgebra;
extern crate ncollide3d;
extern crate nphysics3d;
extern crate obj;
extern crate rayon;
extern crate simdnoise;
extern crate specs;
extern crate sys_info;

#[macro_use]
mod helper;
#[macro_use]
mod logger;
mod config;
mod ecs;
mod graphics;
mod objects;

use config::configloader::{self, UpdateConfig};
use ecs::{components::PitchYawRoll, entities};
use graphics::{
    renderer::{pipelines::Pipeline, *},
    *,
};
use helper::*;
use log::{info, warn};
use logger::*;
use objects::*;
use rayon::ThreadPoolBuilder;
use simdnoise::NoiseType;
use std::time::Instant;
use vulkano::instance::Version;
use winit::{DeviceEvent, Event, KeyboardInput, MouseScrollDelta, WindowEvent};

#[cfg(debug_assertions)]
const DEBUG: bool = true;
#[cfg(not(debug_assertions))]
const DEBUG: bool = false;

const NAME: &str = "Game";
const VERSION: Version = Version {
    major: 0,
    minor: 1,
    patch: 0,
};

fn main() {
    // comment and uncomment to recompile shaders
    assert!(true);

    init();
    Logger::init(&format!("{}.log", NAME));

    let pool = ThreadPoolBuilder::new()
        .build()
        .unwrap_or_else(|err| error_close!("{}", err));
    let mut config = configloader::Config::new("config.toml");
    let mut events_loop = winit::EventsLoop::new();
    let (mut renderer, _debug, device_name) = Renderer::new(&events_loop, &config);
    let objects = Objects::load(&renderer.queue);
    let mut ecs = ecs::init();
    let mut physics = create_physics();
    let player = entities::create_player(&mut ecs, Vec3::new(10.0, 10.0, 125.0), 0.0, 0.0);

    let terrain = objects::gen_terrain(
        &renderer.queue,
        10.0,
        0.0,
        0.0,
        NoiseType::Fbm {
            freq: 0.11,
            lacunarity: 0.5,
            gain: 2.0,
            octaves: 3,
        },
    );

    entities::create_obj(
        &mut ecs,
        Pipeline::terrain,
        terrain,
        Vec3::new(0.0, 0.0, 0.0),
        0.0,
        0.0,
        0.0,
    );

    //entities::create_obj(&mut ecs, Pipeline::normal, Object::suzanne, Vec3::new(0.0, 5.0, 0.0), 0.0, 0.0, 0.0);

    // testing physics
    use nphysics3d::volumetric::Volumetric;

    let cuboid = ncollide3d::shape::ShapeHandle::new(ncollide3d::shape::Cuboid::new(
        nalgebra::Vector3::new(1.0, 2.0, 1.0),
    ));
    let local_inertia = cuboid.inertia(1.0);
    let local_center_of_mass = cuboid.center_of_mass();
    let test = physics.add_rigid_body(
        nphysics3d::math::Isometry::new(nalgebra::Vector3::x() * 2.0, nalgebra::zero()),
        local_inertia,
        local_center_of_mass,
    );

    if DEBUG {
        warn!("This is a debug build, beware of any bugs or issues")
    }
    log_system_info(device_name);
    info!("{} {} - Made by Friz64", NAME, VERSION);

    let mut old_frame = Instant::now();
    let mut running = true;
    while running {
        // update deltatime
        let delta_time = {
            let this_frame = Instant::now();
            let frametime = this_frame.duration_since(old_frame);
            old_frame = this_frame;

            frametime.as_secs() as f32 + frametime.subsec_nanos() as f32 / 1_000_000_000.0
        };

        // handle events
        events_loop.poll_events(|event| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode,
                            state,
                            ..
                        },
                    ..
                } => {
                    config.update_keys(virtual_keycode, state);
                }
                WindowEvent::MouseInput { button, state, .. } => {
                    config.update_mouse(button, state);
                }
                WindowEvent::CloseRequested => running = false,
                WindowEvent::Focused(foc) => renderer.focused = foc,
                _ => (),
            },
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseWheel { delta } if renderer.focused => {
                    if let MouseScrollDelta::LineDelta(_, y) = delta {
                        config.update_scroll(y);
                    }
                }
                DeviceEvent::MouseMotion { delta }
                    if renderer.focused && renderer.cursor_grabbed =>
                {
                    // update player pyr
                    let mut pyr_storage = ecs.write_storage::<PitchYawRoll>();
                    let mut player_pyr = pyr_storage.get_mut(player).unwrap_or_else(|| {
                        ::error_close!("Invalid Player Entity; missing Pos Component")
                    });

                    let mouse_speed = config.controls.sensitivity.mouse_speed * 0.85;

                    player_pyr.0 -= delta.1 as f32 * mouse_speed * 0.005;
                    player_pyr.1 += delta.0 as f32 * mouse_speed * 0.005;

                    let max_pitch = std::f32::consts::FRAC_PI_2 - 0.0001;

                    if player_pyr.0 > max_pitch {
                        // locks looking up
                        player_pyr.0 = max_pitch;
                    } else if player_pyr.0 < -max_pitch {
                        // locks looking down
                        player_pyr.0 = -max_pitch;
                    }
                }
                _ => (),
            },
            _ => (),
        });

        // updates down, hold, up and none
        config.update_status();

        if config.controls.engine.grab_cursor.down() {
            renderer.cursor_grabbed = !renderer.cursor_grabbed;

            renderer
                .surface
                .window()
                .grab_cursor(renderer.cursor_grabbed)
                .unwrap_or_else(|err| ::error_close!("{}", err));

            renderer
                .surface
                .window()
                .hide_cursor(renderer.cursor_grabbed);
        }

        // perform physics
        physics.set_timestep(delta_time);

        //physics.step();

        //println!("pos: {}", physics.rigid_body(test).unwrap().position().translation.vector.y);

        //println!("fps: {}", 1.0 / delta_time);

        // render
        if renderer.draw(delta_time, &ecs, player, &objects, &config) {
            continue;
        };

        // sets scroll back to none
        config.update_scroll(0.0);
    }

    info!("Exiting...");
}
