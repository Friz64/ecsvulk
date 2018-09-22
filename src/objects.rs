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
};

macro_rules! gen_objects {
    ( $( $name:ident ),* ) => {
        #[derive(Debug)]
        pub struct Objects {
            $(
                pub $name: Model,
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
    };
}

impl Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Model - vertices: {} - normals: {} - indices: {}", self.vertex_buf.is_some(), self.normals_buf.is_some(), self.index_buf.is_some())
    }
}

impl Default for Model {
    fn default() -> Self {
        Self {
            vertex_buf: None,
            normals_buf: None,
            index_buf: None,
        }
    }
}

fn load_obj(logger: &mut Logger, queue: &Arc<Queue>, name: &str) -> Option<Model> {
    let path = format!("./{}/models/{}.obj", ::NAME, name);

    let input = BufReader::new(
        match File::open(path) {
            Ok(res) => res,
            Err(err) => {
                logger.warning("OpenOBJ", format!("Failed to open {}: {}", name, err));
                return None;
            },
        }
    );

    let obj: obj::Obj<obj::Vertex> = match obj::load_obj(input) {
        Ok(res) => res,
        Err(err) => {
            logger.warning("LoadOBJ", format!("Failed to load {}: {}", name, err));
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

    Some(Model {
        vertex_buf: Some(Renderer::vertex_buffer(logger, queue, &vertices)),
        normals_buf: Some(Renderer::normals_buffer(logger, queue, &normals)),
        index_buf: Some(Renderer::index_buffer(logger, &queue, &indices)),
    })
}

// actually does the work, specify the objs here
gen_objects!(teapot, a);
