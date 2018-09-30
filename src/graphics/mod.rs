// the pipeline system
#[macro_use]
pub mod pipeline;

// inits vulkan and draws
pub mod renderer;

pub type Vec3 = ::cgmath::Vector3<f32>;
pub type Point3 = ::cgmath::Point3<f32>;
pub type Mat4 = ::cgmath::Matrix4<f32>;

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub pos: [f32; 3],
}
impl_vertex!(Vertex, pos);

#[derive(Copy, Clone, Debug)]
pub struct Normal {
    pub normal: [f32; 3],
}
impl_vertex!(Normal, normal);

pub fn add(first: &mut [f32; 3], other: Vec3) {
    first[0] += other.x;
    first[1] += other.y;
    first[2] += other.z;
}