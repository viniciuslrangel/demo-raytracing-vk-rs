#version 450

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D u_color;
layout(set = 0, binding = 1) uniform sampler2D u_albedo;
layout(set = 0, binding = 2) uniform sampler2D u_normal;
layout(set = 0, binding = 3) uniform sampler2D u_depth;

layout(set = 0, binding = 4) uniform RenderInfo {
    int selected_view;
    int kernel_size;
    int kernel_offset;
    float albedo_weight;
    float normal_weight;
    float depth_weight;
} renderInfo;

void main() {

    int view = renderInfo.selected_view;

    if (view != 0) {
        switch (view) {
            case 1: f_color = texelFetch(u_color, ivec2(gl_FragCoord), 0); break;
            case 2: f_color = vec4(texelFetch(u_albedo, ivec2(gl_FragCoord), 0).rgb, 1.0); break;
            case 3: f_color = vec4(texelFetch(u_normal, ivec2(gl_FragCoord), 0).rgb, 1.0); break;
            case 4: f_color = vec4(vec3(
            texelFetch(u_depth, ivec2(gl_FragCoord), 0).r
            ), 1.0); break;
        }
        return;
    }

    ivec2 coord = ivec2(gl_FragCoord);

    int kernel_width = renderInfo.kernel_size;
    int kernel_height = renderInfo.kernel_size;

    int kernel_offset = renderInfo.kernel_offset;

    float albedo_weight = renderInfo.albedo_weight;
    float normal_weight = renderInfo.normal_weight;
    float depth_weight = renderInfo.depth_weight;

    vec3 color = vec3(0);
    float total_weight = 0.0;

    vec3 center_albedo = texelFetch(u_albedo, coord, 0).rgb;
    vec3 center_normal = texelFetch(u_normal, coord, 0).rgb;
    float center_depth = texelFetch(u_depth, coord, 0).r;

    for (int x = -kernel_width; x <= kernel_width; x += kernel_offset) {
        for (int y = -kernel_height; y <= kernel_height; y += kernel_offset) {

            vec3 c = texelFetch(u_color, coord + ivec2(x, y), 0).rgb;
            vec3 a = texelFetch(u_albedo, coord + ivec2(x, y), 0).rgb;
            vec3 n = texelFetch(u_normal, coord + ivec2(x, y), 0).rgb;
            float d = texelFetch(u_depth, coord + ivec2(x, y), 0).r;

            float weight = 1.0;

            weight *= exp(-dot(a - center_albedo, a - center_albedo) / albedo_weight);
            weight *= exp(-dot(n - center_normal, n - center_normal) / normal_weight);
            weight *= exp(-abs(d - center_depth) / depth_weight);

            color += c * weight;
            total_weight += weight;
        }
    }

    f_color = vec4(color / total_weight, 1.0);
}