use logger::Logger;
use obj;
use std::{
    io::BufReader,
    fs::File,
};
use graphics::renderer::{
    Vertex,
};
use ::graphics::renderer::Vec3;

/*
pub fn load(logger: &mut Logger, name: &str) -> (Vec<Vertex>, Vec<Normal>, Vec<u16>) {
    let path = format!("./{}/models/{}.obj", ::NAME, name);

    let input = BufReader::new(File::open(path).unwrap());
    let obj = obj::load_obj(input).unwrap();

    output
}*/