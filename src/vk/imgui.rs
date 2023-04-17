use std::fmt;
use std::sync::Arc;

use imgui::{DrawCmd, DrawCmdParams, DrawVert, TextureId, Textures};
use imgui::internal::RawWrapper;
use vulkano::buffer::{BufferContents, BufferUsage};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::image::{ImageDimensions, ImmutableImage, MipmapsCount};
use vulkano::image::view::ImageView;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint, StateMode};
use vulkano::pipeline::graphics::color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState, ColorComponents};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Scissor, ViewportState};
use vulkano::render_pass::Subpass;
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};

use crate::vk::vk::Vk;

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/vert_imgui.glsl",
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/frag_imgui.glsl",
    }
}

#[derive(Default, Debug, Clone, BufferContents, vulkano::pipeline::graphics::vertex_input::Vertex)]
#[repr(C)]
struct Vert {
    #[format(R32G32_SFLOAT)]
    pub pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
    #[format(R32_UINT)]
    pub col: u32,
    // pub col: [u8; 4],
}

impl From<DrawVert> for Vert {
    fn from(v: DrawVert) -> Vert {
        unsafe { std::mem::transmute(v) }
    }
}

#[derive(Debug)]
pub enum RendererError {
    BadTexture(TextureId),
    BadImageDimensions(ImageDimensions),
}

impl fmt::Display for RendererError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            &Self::BadTexture(ref t) => {
                write!(f, "The Texture ID could not be found: {:?}", t)
            }
            &Self::BadImageDimensions(d) => {
                write!(f, "Image Dimensions not supported (must be Dim2d): {:?}", d)
            }
        }
    }
}

impl std::error::Error for RendererError {}


pub type Texture = (Arc<ImageView<ImmutableImage>>, Arc<Sampler>);

pub struct ImGuiRenderer {
    pipeline: Arc<GraphicsPipeline>,
    font_texture: Texture,
    textures: Textures<Texture>,
    vertex_allocator: SubbufferAllocator,
    index_allocator: SubbufferAllocator,
}

impl ImGuiRenderer {
    /// Initialize the renderer object, including vertex buffers, ImGui font textures,
    /// and the Vulkan graphics pipeline.
    ///
    /// ---
    ///
    /// `ctx`: the ImGui `Context` object
    pub fn init(
        ctx: &mut imgui::Context,
        vk: &Vk,
        // format: Format,
        render_pass: Subpass,
    ) -> Result<ImGuiRenderer, Box<dyn std::error::Error>> {
        let vs = vs::load(vk.device.clone()).unwrap();
        let fs = fs::load(vk.device.clone()).unwrap();

        /*let render_pass = vulkano::single_pass_renderpass!(
                vk.device.clone(),
                attachments: {
                    color: {
                        load: Load,
                        store: Store,
                        format: format,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            )?;*/

        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(Vert::per_vertex())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_dynamic(1))
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .color_blend_state(ColorBlendState {
                attachments: vec![ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend::alpha()),
                    color_write_mask: ColorComponents::all(),
                    color_write_enable: StateMode::Fixed(true),
                }],
                ..Default::default()
            })
            .render_pass(render_pass)
            .build(vk.device.clone())?;


        let textures = Textures::new();

        let font_texture = Self::upload_font_texture(&vk, ctx.fonts())?;

        ctx.set_renderer_name(format!("imgui-vulkano-renderer {}", env!("CARGO_PKG_VERSION")));

        let vertex_allocator = SubbufferAllocator::new(
            vk.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
        );

        let index_allocator = SubbufferAllocator::new(
            vk.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
        );

        Ok(ImGuiRenderer {
            pipeline,
            font_texture,
            textures,
            vertex_allocator,
            index_allocator,
        })
    }

    /// Appends the draw commands for the UI frame to an `AutoCommandBufferBuilder`.
    ///
    /// ---
    ///
    /// `cmd_buf_builder`: An `AutoCommandBufferBuilder` from vulkano to add commands to
    ///
    /// `device`: the Vulkano `Device` object for the device you want to render the UI on
    ///
    /// `queue`: the Vulkano `Queue` object for buffer creation
    ///
    /// `target`: the target image to render to
    ///
    /// `draw_data`: the ImGui `DrawData` that each UI frame creates
    pub fn draw_commands(
        &mut self,
        cmd_buf_builder:
        &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        vk: &Vk,
        draw_data: &imgui::DrawData,
    ) {
        let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
        if !(fb_width > 0.0 && fb_height > 0.0) {
            return;
        }
        let left = draw_data.display_pos[0];
        let right = draw_data.display_pos[0] + draw_data.display_size[0];
        let top = draw_data.display_pos[1];
        let bottom = draw_data.display_pos[1] + draw_data.display_size[1];

        let pc = vs::VertPC {
            matrix: [
                [(2.0 / (right - left)), 0.0, 0.0, 0.0],
                [0.0, (2.0 / (bottom - top)), 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [
                    (right + left) / (left - right),
                    (top + bottom) / (top - bottom),
                    0.0,
                    1.0,
                ],
            ]
        };

        let clip_off = draw_data.display_pos;
        let clip_scale = draw_data.framebuffer_scale;


        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();

        cmd_buf_builder.bind_pipeline_graphics(self.pipeline.clone());
        for draw_list in draw_data.draw_lists() {

            // let vertex_buffer = Arc::new(self.vrt_buffer_pool.chunk(draw_list.vtx_buffer().iter().map(|&v| Vertex::from(v))).unwrap());
            // let index_buffer = Arc::new(self.idx_buffer_pool.chunk(draw_list.idx_buffer().iter().cloned()).unwrap());

            let vertex_buffer = {
                let buf = draw_list.vtx_buffer();
                let subbuffer = self.vertex_allocator.allocate_slice(buf.len() as u64).unwrap();
                let mut write = subbuffer.write().unwrap();
                for (i, v) in buf.iter().enumerate() {
                    write[i] = Vert::from(*v);
                }
                drop(write);
                subbuffer
            };
            let index_buffer = {
                let buf = draw_list.idx_buffer();
                let subbuffer = self.index_allocator.allocate_slice(buf.len() as u64).unwrap();
                let mut write = subbuffer.write().unwrap();
                for (i, v) in buf.iter().enumerate() {
                    write[i] = *v;
                }
                drop(write);
                subbuffer
            };

            for cmd in draw_list.commands() {
                match cmd {
                    DrawCmd::Elements {
                        count,
                        cmd_params:
                        DrawCmdParams {
                            clip_rect,
                            texture_id,
                            vtx_offset,
                            idx_offset,
                            ..
                        },
                    } => {
                        let clip_rect = [
                            (clip_rect[0] - clip_off[0]) * clip_scale[0],
                            (clip_rect[1] - clip_off[1]) * clip_scale[1],
                            (clip_rect[2] - clip_off[0]) * clip_scale[0],
                            (clip_rect[3] - clip_off[1]) * clip_scale[1],
                        ];

                        if clip_rect[0] < fb_width
                            && clip_rect[1] < fb_height
                            && clip_rect[2] >= 0.0
                            && clip_rect[3] >= 0.0
                        {
                            let scissor = Scissor {
                                origin: [
                                    f32::max(0.0, clip_rect[0]).floor() as u32,
                                    f32::max(0.0, clip_rect[1]).floor() as u32,
                                ],
                                dimensions: [
                                    (clip_rect[2] - clip_rect[0]).abs().ceil() as u32,
                                    (clip_rect[3] - clip_rect[1]).abs().ceil() as u32,
                                ],
                            };

                            cmd_buf_builder.set_scissor(0, [scissor]);

                            let (texture, sampler) = self.lookup_texture(texture_id).unwrap();

                            let set = PersistentDescriptorSet::new(
                                &vk.descriptor_set_allocator,
                                layout.clone(),
                                [
                                    WriteDescriptorSet::image_view_sampler(0, texture.clone(), sampler.clone()),
                                ],
                            ).unwrap();

                            cmd_buf_builder
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Graphics,
                                    self.pipeline.layout().clone(),
                                    0,
                                    set,
                                )
                                .push_constants(
                                    self.pipeline.layout().clone(),
                                    0,
                                    pc,
                                )
                                .bind_vertex_buffers(0, vec![vertex_buffer.clone()])
                                .bind_index_buffer(index_buffer.clone())
                                .draw_indexed(
                                    count as u32,
                                    1,
                                    idx_offset as u32,
                                    vtx_offset as i32,
                                    0,
                                ).unwrap();
                        }
                    }
                    DrawCmd::ResetRenderState => (), // TODO
                    DrawCmd::RawCallback { callback, raw_cmd } => unsafe {
                        callback(draw_list.raw(), raw_cmd)
                    },
                }
            }
        }
    }

    fn upload_font_texture(
        vk: &Vk,
        fonts: &mut imgui::FontAtlas,
    ) -> Result<Texture, Box<dyn std::error::Error>> {
        let texture = fonts.build_rgba32_texture();

        let mut upload = vk.uploads.as_ref().unwrap().borrow_mut();
        let image = ImmutableImage::from_iter(
            &vk.memory_allocator,
            texture.data.iter().copied(),
            ImageDimensions::Dim2d {
                width: texture.width,
                height: texture.height,
                array_layers: 1,
            },
            MipmapsCount::One,
            Format::R8G8B8A8_SRGB,
            &mut upload,
        ).unwrap();
        drop(upload);

        let texture = ImageView::new_default(image).unwrap();

        let sampler = Sampler::new(
            vk.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        ).unwrap();

        fonts.tex_id = TextureId::from(usize::MAX);
        Ok((texture, sampler))
    }

    fn lookup_texture(&self, texture_id: TextureId) -> Result<&Texture, RendererError> {
        if texture_id.id() == usize::MAX {
            Ok(&self.font_texture)
        } else if let Some(texture) = self.textures.get(texture_id) {
            Ok(texture)
        } else {
            Err(RendererError::BadTexture(texture_id))
        }
    }
}
