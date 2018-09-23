/*
Components:
- Position
- Model
- Controls
*/
use ::std::{
    sync::Arc,
};
use ::specs::{
    prelude::*,
    storage::{HashMapStorage},
};
use ::graphics::renderer::{
    Vec3,
};
use ::objects::Object;

pub struct Pos(pub Vec3);
impl Component for Pos {
    type Storage = VecStorage<Self>;
}

pub struct PitchYawRoll(pub f32, pub f32, pub f32);
impl Component for PitchYawRoll {
    type Storage = VecStorage<Self>;
}

pub struct Model(pub Object);
impl Component for Model {
    type Storage = VecStorage<Self>;
}

pub struct Controls;
impl Default for Controls {
    fn default() -> Self { Controls }
}
impl Component for Controls {
    type Storage = NullStorage<Self>;
}