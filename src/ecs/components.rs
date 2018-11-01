use ::specs::{
    prelude::*,
    storage::{HashMapStorage},
};
use ::graphics::Vec3;
use ::objects::Object;
use ::renderer::pipelines;
use ::nphysics3d::object::BodyHandle;

pub struct Pos(pub Vec3);
impl Component for Pos {
    type Storage = VecStorage<Self>;
}

pub struct PitchYawRoll(pub f32, pub f32, pub f32);
impl Component for PitchYawRoll {
    type Storage = VecStorage<Self>;
}

pub struct SpeedMultiplier(pub f32);
impl Component for SpeedMultiplier {
    type Storage = HashMapStorage<Self>;
}

pub struct Wireframe(pub bool);
impl Component for Wireframe {
    type Storage = HashMapStorage<Self>;
}

pub struct Model(pub Object);
impl Component for Model {
    type Storage = VecStorage<Self>;
}

pub struct Pipeline(pub pipelines::Pipeline);
impl Component for Pipeline {
    type Storage = VecStorage<Self>;
}

pub struct Physics(pub BodyHandle);
impl Component for Physics {
    type Storage = VecStorage<Self>;
}