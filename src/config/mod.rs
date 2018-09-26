// contains macro for config generation
#[macro_use]
pub mod configgen;

// contains user input logic
pub mod keycode;

// calls configgen macro and adds additional logic
pub mod configloader;