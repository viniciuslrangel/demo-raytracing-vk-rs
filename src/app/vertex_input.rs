use vulkano::buffer::BufferContents;

#[derive(BufferContents, vulkano::pipeline::graphics::vertex_input::Vertex)]
#[repr(C)]
pub struct ScreenVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
}
