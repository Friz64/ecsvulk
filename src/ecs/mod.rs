pub mod components;
pub mod systems;
pub mod entities;

use self::components::*;
//use self::systems::*;

use ::specs::{
    World,
};

pub fn init() -> World {
    let mut world = World::new();

    world.register::<Pos>();
    world.register::<PitchYawRoll>();
    world.register::<Model>();
    world.register::<Controls>();

    world
}