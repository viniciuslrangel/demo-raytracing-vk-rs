#version 450

// raytracing vertex shader

layout(location = 0) in vec2 position;

layout(location = 0) out vec2 coord;

void main() {
//    gl_Position = viewData.proj * viewData.worldview * vec4(position, 0.0, 1.0);
    coord = position * -1.0;
    gl_Position = vec4(position, 0.0, 1.0);
}