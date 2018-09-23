use logger::Logger;
use obj;
use std::{
    io::BufReader,
    fs::File,
    fmt::{self, Debug},
    default::Default,
    sync::Arc,
};
use ecs::components::Model;
use graphics::renderer::{
    Vertex, Normal, Renderer,
};
use ::vulkano::{
    device::Queue,
    buffer::{BufferAccess, TypedBufferAccess},
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
                Self {
                    $(
                        $name: load_obj(logger, queue, stringify!($name)).unwrap_or_default(),
                    )*
                }
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
        index_buf: Some(Renderer::index_buffer(logger, &queue, &indices)),
    })
}

// actually does the work, specify the objs here
gen_objects!(teapot);
