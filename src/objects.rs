use logger::Logger;
use obj;
use std::{
    io::BufReader,
    fs::File,
    default::Default,
    sync::Arc,
};
use graphics::renderer::{
    Vertex, Normal, Renderer, Vec3,
};
use ::vulkano::{
    device::Queue,
    buffer::{BufferAccess, TypedBufferAccess},
};
use ::simdnoise::{
    self, NoiseType,
};
use ::cgmath::{
    InnerSpace,
};

pub struct ModelBuffers {
    vertex_buf: Option<Arc<BufferAccess + Send + Sync>>,
    normals_buf: Option<Arc<BufferAccess + Send + Sync>>,
    index_buf: Option<Arc<TypedBufferAccess<Content=[u16]> + Send + Sync>>,
}

macro_rules! gen_objects {
    ( $( $name:ident ),* ) => {
        pub struct Objects {
            $(
                pub $name: ModelBuffers,
            )*
        }

        impl Objects {
            pub fn load(logger: &mut Logger, queue: &Arc<Queue>) -> Self {
                let objs = Self {
                    $(
                        $name: load_obj(logger, queue, stringify!($name)).unwrap_or_default(),
                    )*
                };

                logger.info("ObjLoader", "Objs loaded");

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

fn load_obj(logger: &mut Logger, queue: &Arc<Queue>, name: &str) -> Option<ModelBuffers> {
    let path = format!("./{}/models/{}.obj", ::NAME, name);

    let input = BufReader::new(
        match File::open(path) {
            Ok(res) => res,
            Err(err) => {
                logger.warning("OpenOBJ", format!("Failed to open {}.obj: {}", name, err));
                return None;
            },
        }
    );

    let obj: obj::Obj<obj::Vertex> = match obj::load_obj(input) {
        Ok(res) => res,
        Err(err) => {
            logger.warning("LoadOBJ", format!("Failed to load {}.obj: {}", name, err));
            return None;
        },
    };

    let mut vertices = vec![];
    let mut normals = vec![];
    obj.vertices.iter()
        .for_each(|vert| {
            vertices.push(Vertex { pos: vert.position });
            normals.push(Normal { normal: vert.normal });
        });;
    let indices = obj.indices;

    Some(ModelBuffers {
        vertex_buf: Some(Renderer::vertex_buffer(logger, queue, &vertices)),
        normals_buf: Some(Renderer::normals_buffer(logger, queue, &normals)),
        index_buf: Some(Renderer::index_buffer(logger, queue, &indices)),
    })
}

// actually does the work, specify the objs here
gen_objects!(teapot, suzanne);

pub fn gen_terrain(logger: &mut Logger, queue: &Arc<Queue>) -> Object {
    #[allow(non_snake_case)]
    let KANTENLAENGE: usize = 256;
    let range: usize = KANTENLAENGE * KANTENLAENGE;

    let noise_type = NoiseType::Normal {
        freq: 0.05,
    };

    let noise = simdnoise::get_2d_scaled_noise(0.0, KANTENLAENGE, 0.0, KANTENLAENGE, noise_type, 0.0, 10.0);

    let mut vertices = vec![];
    let mut normals = vec![];
    for i in 0..range {
        vertices.push(Vertex { pos: [(i / KANTENLAENGE) as f32, noise[i], (i % KANTENLAENGE) as f32]});
        normals.push(Normal { normal: [0.0, 0.0, 0.0] });
    }

    let mut indices = vec![];
    for i in 0..range - (KANTENLAENGE * 2 - 1) {
        let x = i / (KANTENLAENGE - 1);
        let z = i % (KANTENLAENGE - 1);

        // triangle 1
        indices.push((x * KANTENLAENGE + z) as u16);
        indices.push(((x * KANTENLAENGE + z) + 1) as u16);
        indices.push(((x + 1) * KANTENLAENGE + z) as u16);

        // triangle 2
        indices.push(((x * KANTENLAENGE + z) + 1) as u16);
        indices.push(((x + 1) * KANTENLAENGE + z + 1) as u16);
        indices.push(((x + 1) * KANTENLAENGE + z) as u16);
    }

    for i in 0..indices.len() / 3 {
        /*
        normal(vec3 a, vec3 b, vec3 c):
        normal = cross(b - a, c - a).normalize()
        a, b, c = vertex position
        */

        let ia = indices[i * 3] as usize;
        let ib = indices[i * 3 + 1] as usize;
        let ic = indices[i * 3 + 2] as usize;

        let a: Vec3 = vertices[ia].pos.into();
        let b: Vec3 = vertices[ib].pos.into();
        let c: Vec3 = vertices[ic].pos.into();

        let ba: Vec3 = b - a;
        let ca: Vec3 = c - a;
        let norm: Vec3 = ba.cross(ca).normalize();

        let normals_ia: Vec3 = normals[ia].normal.into();
        normals[ia].normal = (normals_ia + norm).into();

        let normals_ib: Vec3 = normals[ib].normal.into();
        normals[ib].normal = (normals_ib + norm).into();

        let normals_ic: Vec3 = normals[ic].normal.into();
        normals[ic].normal = (normals_ic + norm).into();
    }

    for i in 0..range {
        let normals_i: Vec3 = normals[i].normal.into();
        normals[i].normal = normals_i.normalize().into()
    }

    Object::Custom(ModelBuffers {
        vertex_buf: Some(Renderer::vertex_buffer(logger, queue, &vertices)),
        normals_buf: Some(Renderer::normals_buffer(logger, queue, &normals)),
        index_buf: Some(Renderer::index_buffer(logger, queue, &indices)),
    })
}