#version 450
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 v_normal;

layout(set = 0, binding = 0) uniform Data {
    mat4 mvp;
    //mat4 view;
    //mat4 model;
} uniforms;

void main() {
    gl_Position = uniforms.mvp * vec4(pos, 1);
    v_normal = normal;
}