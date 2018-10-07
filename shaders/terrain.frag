#version 450

layout(location = 0) out vec4 f_color;

layout(location = 0) in vec3 v_pos;
layout(location = 1) in vec3 v_normal;

void main() {
    vec3 color = v_normal;

    if (v_pos.y < 2.75) {
        color = vec3(0.0, 0.12, 0.58) * (1 - (v_pos.y - 2.75) / 2.75);
    }

    f_color = vec4(color, 1.0);
}