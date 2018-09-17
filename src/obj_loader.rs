use logger::Logger;
use obj::{Obj, SimplePolygon};
use std::path::Path;
use graphics::renderer::Vertex;
use ::graphics::renderer::Vec3;

fn load(logger: &mut Logger, path: &str) -> Vec<Vertex> {
    let obj = match Obj::<SimplePolygon>::load(Path::new(path)) {
        Ok(res) => res,
        Err(err) => {
            logger.warning("LoadObj", format!("IGNORING {}: {}", path, err));
            return vec![];
        },
    };
    
    obj.position.iter()
        .zip(obj.normal.iter())
        .map(|(pos, normal)| Vertex::new(Vec3::from(*pos), Vec3::from(*normal)))
        .collect()
}