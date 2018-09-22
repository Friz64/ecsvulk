/*
Components:
- Position
- Model
- Controls
*/
use ::std::{
    sync::Arc,
};
use ::vulkano::{
    buffer::{BufferAccess, TypedBufferAccess},
};
use ::specs::{
    prelude::*,
    storage::{HashMapStorage},
};
use ::graphics::renderer::{
    Vec3, Vertex, Normal,
};

struct Pos(Vec3);
impl Component for Pos {
    type Storage = VecStorage<Self>;
}

pub struct Model {
    pub vertex_buf: Option<Arc<BufferAccess + Send + Sync>>,
    pub normals_buf: Option<Arc<BufferAccess + Send + Sync>>,
    pub index_buf: Option<Arc<TypedBufferAccess<Content=[u16]> + Send + Sync>>,
}
impl Component for Model {
    type Storage = VecStorage<Self>;
}