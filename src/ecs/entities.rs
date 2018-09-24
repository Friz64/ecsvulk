use super::components::*;
use ::graphics::renderer::{
    Vec3
};
use ::objects::{
    Object,
};
use specs::{
    World, Builder, Entity,
};

pub fn create_player(world: &mut World, pos: Vec3, pitch: f32, yaw: f32) -> Entity {
    world.create_entity()
        .with(Pos(pos))
        .with(PitchYawRoll(pitch, yaw, 0.0))
        .with(SpeedMultiplier(1.0))
        .build()
}

pub fn create_obj(world: &mut World, object: Object, pos: Vec3, pitch: f32, yaw: f32, roll: f32) -> Entity {
    world.create_entity()
        .with(Pos(pos))
        .with(PitchYawRoll(pitch, yaw, roll))
        .with(Model(object))
        .build()
}