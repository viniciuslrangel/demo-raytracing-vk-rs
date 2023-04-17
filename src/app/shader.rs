pub mod raytrace {
    pub mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shaders/vert_raytracing.glsl",
        }
    }
    pub mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shaders/frag_raytracing.glsl",
        }
    }
}
pub mod denoiser {
    pub mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shaders/vert_denoiser.glsl",
        }
    }
    pub mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shaders/frag_denoiser.glsl",
        }
    }
}