pub mod components;
pub mod resources;
pub mod systems;
pub mod entities;

use self::components::*;
//use self::systems::*;

use ::specs::World;
use ::std::collections::HashMap;

pub fn init() -> World {
    let mut world = World::new();

    world.register::<Pos>();
    world.register::<PitchYawRoll>();
    world.register::<Model>();
    world.register::<SpeedMultiplier>();
    world.register::<Wireframe>();
    world.register::<Pipeline>();

    //world.add_resource::<HashMap<>>(HashMap::new());

    world
}

pub fn maintain(world: &mut World) {
    world.maintain();
}