use cgmath::InnerSpace;
use graphics::{self, renderer::Renderer, Normal, Vec3, Vertex};
use log::{info, warn};
use obj;
use simdnoise::{self, NoiseType};
use std::{default::Default, fs::File, io::BufReader, sync::Arc};
use vulkano::{
    buffer::{BufferAccess, TypedBufferAccess},
    device::Queue,
};

pub struct ModelBuffers {
    vertex_buf: Option<Arc<BufferAccess + Send + Sync>>,
    normals_buf: Option<Arc<BufferAccess + Send + Sync>>,
    index_buf: Option<Arc<TypedBufferAccess<Content = [u16]> + Send + Sync>>,
}

macro_rules! gen_objects {
    ( $( $name:ident ),* ) => {
        pub struct Objects {
            $(
                pub $name: ModelBuffers,
            )*
        }

        impl Objects {
            pub fn load(queue: &Arc<Queue>) -> Self {
                let objs = Self {
                    $(
                        $name: load_obj(queue, stringify!($name)).unwrap_or_default(),
                    )*
                };

                info!("Objs loaded");

                objs
            }
        }

        #[allow(non_camel_case_types)]
        pub enum Object {
            $(
                $name,
            )*
            Custom(ModelBuffers)
        }

        impl Object {
            pub fn get_buffers(&self, objects: &Objects) -> Option<(
                Arc<BufferAccess + Send + Sync>,
                Arc<BufferAccess + Send + Sync>,
                Arc<TypedBufferAccess<Content=[u16]> + Send + Sync>,
            )> {
                match self {
                    $(
                        Object::$name => {
                            if let Some(ref vertex_buf) = objects.$name.vertex_buf {
                                if let Some(ref normals_buf) = objects.$name.normals_buf {
                                    if let Some(ref index_buf) = objects.$name.index_buf {
                                        return Some((vertex_buf.clone(), normals_buf.clone(), index_buf.clone()));
                                    }
                                }
                            }

                            None
                        },
                    )*
                    Object::Custom(buffers) => {
                        if let Some(ref vertex_buf) = buffers.vertex_buf {
                            if let Some(ref normals_buf) = buffers.normals_buf {
                                if let Some(ref index_buf) = buffers.index_buf {
                                    return Some((vertex_buf.clone(), normals_buf.clone(), index_buf.clone()));
                                }
                            }
                        }

                        None
                    }
                }
            }
        }
    };
}

impl Default for ModelBuffers {
    fn default() -> Self {
        Self {
            vertex_buf: None,
            normals_buf: None,
            index_buf: None,
        }
    }
}

fn load_obj(queue: &Arc<Queue>, name: &str) -> Option<ModelBuffers> {
    let path = format!("./{}/models/{}.obj", ::NAME, name);

    let input = BufReader::new(match File::open(path) {
        Ok(res) => res,
        Err(err) => {
            warn!("Failed to open {}.obj: {}", name, err);
            return None;
        }
    });

    let obj: obj::Obj<obj::Vertex> = match obj::load_obj(input) {
        Ok(res) => res,
        Err(err) => {
            warn!("Failed to load {}.obj: {}", name, err);
            return None;
        }
    };

    let mut vertices = vec![];
    let mut normals = vec![];
    obj.vertices.iter().for_each(|vert| {
        vertices.push(Vertex { pos: vert.position });
        normals.push(Normal {
            normal: vert.normal,
        });
    });;
    let indices = obj.indices;

    Some(ModelBuffers {
        vertex_buf: Some(Renderer::vertex_buffer(queue, &vertices)),
        normals_buf: Some(Renderer::normals_buffer(queue, &normals)),
        index_buf: Some(Renderer::index_buffer(queue, &indices)),
    })
}

// actually does the work, specify the objs here
gen_objects!(suzanne);

pub fn gen_terrain(
    queue: &Arc<Queue>,
    scale: f32,
    x_off: f32,
    y_off: f32,
    noise_type: NoiseType,
) -> Object {
    let length: usize = 255;
    let range = length.pow(2);

    let noise =
        simdnoise::get_2d_scaled_noise(x_off, length, y_off, length, noise_type, 0.0, scale);

    let mut vertices = vec![];
    let mut normals = vec![];
    let mut indices = vec![];

    // iterates over every point, generates a vertex and a default normal
    for (i, noise_y) in noise.iter().enumerate().take(range) {
        vertices.push(Vertex {
            pos: [(i / length) as f32, *noise_y, (i % length) as f32],
        });
        normals.push(Normal {
            normal: [0.0, 0.0, 0.0],
        });
    }

    // generate indices
    for i in 0..range - (length * 2 - 1) {
        // gets x and z from i
        let x = i / (length - 1);
        let z = i % (length - 1);

        // triangle 1
        indices.push((x * length + z) as u16);
        indices.push(((x * length + z) + 1) as u16);
        indices.push(((x + 1) * length + z) as u16);

        // triangle 2
        indices.push(((x * length + z) + 1) as u16);
        indices.push(((x + 1) * length + z + 1) as u16);
        indices.push(((x + 1) * length + z) as u16);
    }

    // generate normals
    for i in 0..indices.len() / 3 {
        // assuming vertices at points a, b, c

        // indices of the points
        let index_a = indices[i * 3] as usize;
        let index_b = indices[i * 3 + 1] as usize;
        let index_c = indices[i * 3 + 2] as usize;

        // positions of the points
        let vertex_a: Vec3 = vertices[index_a].pos.into();
        let vertex_b: Vec3 = vertices[index_b].pos.into();
        let vertex_c: Vec3 = vertices[index_c].pos.into();

        // magic https://computergraphics.stackexchange.com/questions/4031/programmatically-generating-vertex-normals
        let normal = (vertex_b - vertex_a).cross(vertex_c - vertex_a);

        // add the normal to the normals of the points
        graphics::add(&mut normals[index_a].normal, normal);
        graphics::add(&mut normals[index_b].normal, normal);
        graphics::add(&mut normals[index_c].normal, normal);
    }

    // normalize every normal
    for graphics_normal in normals.iter_mut().take(range) {
        let normal: Vec3 = graphics_normal.normal.into();
        graphics_normal.normal = normal.normalize().into();
    }

    let result = Object::Custom(ModelBuffers {
        vertex_buf: Some(Renderer::vertex_buffer(queue, &vertices)),
        normals_buf: Some(Renderer::normals_buffer(queue, &normals)),
        index_buf: Some(Renderer::index_buffer(queue, &indices)),
    });

    info!("Terrain generated with length {}", length);

    result
}
