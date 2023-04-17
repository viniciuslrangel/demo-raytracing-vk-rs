use std::cell::RefCell;
use std::cmp::max;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use imgui::Context;
use imgui::Ui;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::Subpass;
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano_win::create_surface_from_winit;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::EventLoop;
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::{Window, WindowBuilder};

use crate::app::camera::Camera;
use crate::app::geom::Circle;
use crate::app::material::Material;
use crate::app::shader;
use crate::app::vertex_input::ScreenVertex;
use crate::imgui_winit_support::{HiDpiMode, WinitPlatform};
use crate::vk::imgui::ImGuiRenderer;
use crate::vk::vk::{DrawStatus, Vk};

#[derive(Default)]
pub struct Scene {
    pub camera: Camera,
    pub all_materials: Vec<Material>,
    pub all_circles: Vec<Circle>,

    pub sample_count: u32,

    pub current_view: i32,
    pub kernel_size: i32,
    pub kernel_offset: i32,
    pub denoiser_albedo_weight: f32,
    pub denoiser_normal_weight: f32,
    pub denoiser_depth_weight: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Info {
    pub time: f32,
    pub fps: f32,
}

pub struct App<F>
{
    window: Arc<Window>,
    event_loop: Option<EventLoop<()>>,

    size: [u32; 2],
    recreate_swapchain: bool,

    run_ui: F,
    scene: Scene,

    pub vulkan: Vk,
    viewport: Viewport,
    raytracing_pipeline: Arc<GraphicsPipeline>,
    denoiser_pipeline: Arc<GraphicsPipeline>,
    sampler: Arc<Sampler>,
    vertex_buffer: Subbuffer<[ScreenVertex]>,

    material_buffer: Option<Rc<RefCell<Subbuffer<shader::raytrace::fs::MaterialBuffer>>>>,
    material_buffer_size: usize,
    circle_buffer: Option<Rc<RefCell<Subbuffer<shader::raytrace::fs::CircleBuffer>>>>,
    circle_buffer_size: usize,

    geom_set: Option<Arc<PersistentDescriptorSet>>,

    imgui: Context,
    imgui_platform: WinitPlatform,
    imgui_renderer: ImGuiRenderer,

    start_time: Instant,
    info: Info,
    pressed_keys: [bool; 165],
}

impl<F> App<F>
    where F: FnMut(&mut bool, &mut Ui, &mut Scene, Info) + 'static
{
    pub fn create(run_ui: F) -> Self
    {
        let default_window_size = [1200, 900];

        let vk_instance = Vk::create_instance();

        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title("Raytracing Demo")
            .with_inner_size(PhysicalSize::new(
                default_window_size[0] as f32,
                default_window_size[1] as f32,
            ))
            .build(&event_loop)
            .expect("Failed to create window");
        let window = Arc::new(window);

        let surface = create_surface_from_winit(window.clone(), vk_instance.clone())
            .expect("Failed to create surface");

        let mut vulkan = Vk::create_device(vk_instance, surface);

        let raytracing_subpass = Subpass::from(vulkan.raytrace_render_pass.clone(), 0).unwrap();
        let raytracing_pipeline = vulkan.create_pipeline(
            raytracing_subpass.clone(),
            ScreenVertex::per_vertex(),
            shader::raytrace::vs::load(vulkan.device.clone()).unwrap()
                .entry_point("main").unwrap(),
            shader::raytrace::fs::load(vulkan.device.clone()).unwrap()
                .entry_point("main").unwrap(),
            raytracing_subpass.num_color_attachments(),
        );

        let denoiser_subpass = Subpass::from(vulkan.screen_render_pass.clone(), 0).unwrap();
        let denoiser_pipeline = vulkan.create_pipeline(
            denoiser_subpass.clone(),
            ScreenVertex::per_vertex(),
            shader::denoiser::vs::load(vulkan.device.clone()).unwrap()
                .entry_point("main").unwrap(),
            shader::denoiser::fs::load(vulkan.device.clone()).unwrap()
                .entry_point("main").unwrap(),
            denoiser_subpass.num_color_attachments(),
        );

        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [0.0, 0.0],
            depth_range: 0.0..1.0,
        };
        vulkan.setup_framebuffer(&mut viewport);

        // quad filling the screen
        let vertices = [
            ScreenVertex { position: [-1.0, -1.0] },
            ScreenVertex { position: [-1.0, 1.0] },
            ScreenVertex { position: [1.0, -1.0] },
            ScreenVertex { position: [1.0, 1.0] },
            ScreenVertex { position: [1.0, -1.0] },
            ScreenVertex { position: [-1.0, 1.0] },
        ];
        let vertex_buffer = Buffer::from_iter(
            &vulkan.memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vertices,
        ).unwrap();

        let sampler = Sampler::new(
            vulkan.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        ).unwrap();

        let mut imgui = Context::create();
        imgui.set_ini_filename(None);

        let mut imgui_platform = WinitPlatform::init(&mut imgui);
        imgui_platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Rounded);

        let imgui_renderer = ImGuiRenderer::init(
            &mut imgui,
            &vulkan,
            // vulkan.swapchain.image_format(),
            denoiser_subpass.clone(),
        ).unwrap();

        vulkan.do_upload();

        let mut camera = Camera::new();
        camera.position[2] = -3.0;

        let scene = Scene {
            camera,
            sample_count: 8,
            current_view: 0,
            kernel_size: 5,
            kernel_offset: 2,
            denoiser_albedo_weight: 0.01,
            denoiser_normal_weight: 0.01,
            denoiser_depth_weight: 0.3,
            ..Default::default()
        };

        Self {
            window,
            event_loop: Some(event_loop),

            size: default_window_size,
            recreate_swapchain: true,

            run_ui,
            scene,

            vulkan,
            viewport,
            raytracing_pipeline,
            denoiser_pipeline,
            sampler,
            vertex_buffer,

            material_buffer: Default::default(),
            material_buffer_size: 0,
            circle_buffer: Default::default(),
            circle_buffer_size: 0,

            geom_set: None,

            imgui,
            imgui_platform,
            imgui_renderer,

            start_time: Instant::now(),
            info: Default::default(),
            pressed_keys: [false; 165],
        }
    }

    fn check_buffers(&mut self) {
        let mut update_descriptors = false;

        let mut recreate_buffer = |s: &mut Self| {
            s.material_buffer = Some(Rc::new(RefCell::new(
                s.vulkan.storage_buffer.allocate_unsized(s.material_buffer_size.clone() as u64).unwrap()
            )));
            s.scene.all_materials.iter_mut().for_each(|m| m.dirty = true);
            update_descriptors = true;
        };
        if self.scene.all_materials.len() != self.material_buffer_size {
            self.material_buffer_size = self.scene.all_materials.len();
            recreate_buffer(self);
        }

        let material_length = self.scene.all_materials.len();
        for i in 0..material_length {
            let m = self.scene.all_materials[i];
            if m.dirty {
                self.scene.all_materials[i].dirty = false;
                let writer = self.material_buffer.as_ref().cloned().unwrap().clone();
                let writer = writer.borrow_mut();
                let writer = writer.write();
                if writer.is_ok() {
                    let mut w = writer.unwrap();
                    w.list[i] = m.into();
                } else {
                    recreate_buffer(self);
                    let writer = self.material_buffer.as_ref().cloned().unwrap().clone();
                    let writer = writer.borrow_mut();
                    let mut w = writer.write().unwrap();
                    w.list[i] = m.into();
                }
            }
        }

        let mut recreate_buffer = |s: &mut Self| {
            s.circle_buffer = Some(Rc::new(RefCell::new(
                s.vulkan.storage_buffer.allocate_unsized(s.circle_buffer_size.clone() as u64).unwrap()
            )));
            s.scene.all_circles.iter_mut().for_each(|c| c.dirty = true);
            update_descriptors = true;
        };

        let circle_length = self.scene.all_circles.len();
        if circle_length != self.circle_buffer_size {
            self.circle_buffer_size = circle_length;
            recreate_buffer(self);
        }

        for i in 0..circle_length {
            let c = self.scene.all_circles[i];
            if c.dirty {
                self.scene.all_circles[i].dirty = false;
                let writer = self.circle_buffer.as_ref().cloned().unwrap().clone();
                let writer = writer.borrow_mut();
                let writer = writer.write();
                if writer.is_ok() {
                    let mut w = writer.unwrap();
                    w.list[i] = c.into();
                } else {
                    recreate_buffer(self);
                    let writer = self.circle_buffer.as_ref().cloned().unwrap().clone();
                    let writer = writer.borrow_mut();
                    let mut w = writer.write().unwrap();
                    w.list[i] = c.into();
                }
            }
        }

        if update_descriptors {
            if let Some(layout) = self.raytracing_pipeline.layout().set_layouts().get(1) {
                let mut descriptor_set = Vec::new();
                if let Some(m) = self.material_buffer.clone() {
                    let buf = m.borrow().clone();
                    descriptor_set.push(WriteDescriptorSet::buffer(0, buf));
                }
                if let Some(c) = self.circle_buffer.clone() {
                    let buf = c.borrow().clone();
                    descriptor_set.push(WriteDescriptorSet::buffer(1, buf));
                }
                let geom_set = PersistentDescriptorSet::new(
                    &self.vulkan.descriptor_set_allocator,
                    layout.clone(),
                    descriptor_set,
                ).unwrap();

                self.geom_set = Some(geom_set);
            }
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.size = [width, height];
        self.recreate_swapchain = true;
        self.scene.camera.set_perspective(75.0, height as f32 / width as f32, 0.1, 100.0);
    }

    pub fn update(&mut self, delta: f32) {
        let mut mov_x = 0_f32;
        let mut mov_y = 0_f32;
        let mut mov_z = 0_f32;
        if self.pressed_keys[VirtualKeyCode::W as usize] {
            mov_z -= 1_f32;
        }
        if self.pressed_keys[VirtualKeyCode::S as usize] {
            mov_z += 1_f32;
        }
        if self.pressed_keys[VirtualKeyCode::A as usize] {
            mov_x -= 1_f32;
        }
        if self.pressed_keys[VirtualKeyCode::D as usize] {
            mov_x += 1_f32;
        }
        if self.pressed_keys[VirtualKeyCode::E as usize] {
            mov_y -= 1_f32;
        }
        if self.pressed_keys[VirtualKeyCode::Q as usize] {
            mov_y += 1_f32;
        }
        if mov_x != 0_f32 || mov_y != 0_f32 || mov_z != 0_f32 {
            self.scene.camera.move_by(mov_x, mov_y, mov_z, delta);
        }
    }

    pub fn main_loop(&mut self) {
        let mut last_frame = Instant::now();

        self.event_loop.take().unwrap().run_return(|event, _, control_flow| {
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    window_id,
                } if window_id == self.window.id() => {
                    control_flow.set_exit();
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(new_size),
                    ..
                } => {
                    self.resize(new_size.width, new_size.height);
                    self.imgui_platform.handle_event(self.imgui.io_mut(), &self.window, &event);
                }
                Event::WindowEvent {
                    event: WindowEvent::KeyboardInput { input, .. },
                    ..
                } => {
                    if let Some(keycode) = input.virtual_keycode {
                        self.pressed_keys[keycode as usize] = input.state == ElementState::Pressed;
                    }
                    self.imgui_platform.handle_event(self.imgui.io_mut(), &self.window, &event);
                }
                Event::MainEventsCleared => {
                    if self.recreate_swapchain {
                        self.recreate_swapchain = false;
                        self.vulkan.recreate_swapchain(self.size, &mut self.viewport);
                    }

                    let now = Instant::now();
                    let delta = now.duration_since(last_frame).as_secs_f32();
                    last_frame = now;

                    self.update(delta);
                    self.info.time = now.duration_since(self.start_time).as_secs_f32();
                    self.info.fps = 1.0 / delta;

                    self.imgui_platform
                        .prepare_frame(self.imgui.io_mut(), &self.window)
                        .expect("Failed to start frame");

                    self.window.request_redraw();
                }
                Event::RedrawRequested(_) => {
                    if self.size[0] == 0 || self.size[1] == 0 {
                        return;
                    }

                    self.check_buffers();

                    self.vulkan.wait_frame();

                    let mut imgui_ui = self.imgui.frame();
                    {
                        let mut run = true;
                        (self.run_ui)(&mut run, &mut imgui_ui, &mut self.scene, self.info);
                        if !run {
                            control_flow.set_exit();
                        }
                    }

                    self.imgui_platform.prepare_render(&imgui_ui, &self.window);
                    let imgui_draw_data = self.imgui.render();

                    let mut render_pass = self.vulkan.begin_frame();
                    if render_pass.is_some() {
                        let render_pass = render_pass.as_mut().unwrap();

                        let view_set = {
                            let view_buffer = {
                                self.scene.camera.update_view();
                                let view_data: shader::raytrace::fs::ViewData = (&self.scene.camera).into();
                                let subbuffer = self.vulkan.uniform_buffer.allocate_sized().unwrap();
                                *subbuffer.write().unwrap() = view_data;
                                subbuffer
                            };

                            let render_info_buffer = {
                                let render_data = shader::raytrace::fs::RenderInfo {
                                    time: self.info.time,
                                    sample_count: self.scene.sample_count as i32,
                                };
                                let subbuffer = self.vulkan.uniform_buffer.allocate_sized().unwrap();
                                *subbuffer.write().unwrap() = render_data;
                                subbuffer
                            };

                            let layout = self.raytracing_pipeline.layout().set_layouts().get(0).unwrap();
                            PersistentDescriptorSet::new(
                                &self.vulkan.descriptor_set_allocator,
                                layout.clone(),
                                [
                                    WriteDescriptorSet::buffer(0, view_buffer),
                                    WriteDescriptorSet::buffer(1, render_info_buffer),
                                ],
                            ).unwrap()
                        };

                        render_pass
                            .set_viewport(0, [self.viewport.clone()])
                            .bind_pipeline_graphics(self.raytracing_pipeline.clone())
                            .bind_vertex_buffers(0, self.vertex_buffer.clone())
                            .bind_descriptor_sets(
                                PipelineBindPoint::Graphics,
                                self.raytracing_pipeline.layout().clone(),
                                0,
                                view_set,
                            );
                        if let Some(geom_set) = self.geom_set.as_ref() {
                            render_pass
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Graphics,
                                    self.raytracing_pipeline.layout().clone(),
                                    1,
                                    geom_set.clone(),
                                );
                        }
                        render_pass
                            .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
                            .unwrap();


                        // END RAYTRACING RENDER_PASS
                        // START DENOISER RENDER_PASS

                        let buffers = self.vulkan.next_render_pass(render_pass);
                        let ray_color = buffers.ray_color_image.clone();
                        let ray_albedo = buffers.ray_albedo_image.clone();
                        let ray_normal = buffers.ray_normal_image.clone();
                        let ray_depth = buffers.ray_depth_image.clone();

                        let render_info = {
                            let render_data = shader::denoiser::fs::RenderInfo {
                                selected_view: self.scene.current_view,
                                kernel_size: self.scene.kernel_size,
                                kernel_offset: max(1, self.scene.kernel_offset),
                                albedo_weight: self.scene.denoiser_albedo_weight,
                                normal_weight: self.scene.denoiser_normal_weight,
                                depth_weight: self.scene.denoiser_depth_weight,
                            };
                            let subbuffer = self.vulkan.uniform_buffer.allocate_sized().unwrap();
                            *subbuffer.write().unwrap() = render_data;
                            subbuffer
                        };

                        let denoiser_descriptor_set = {
                            let layout = self.denoiser_pipeline.layout().set_layouts().get(0).unwrap();
                            PersistentDescriptorSet::new(
                                &self.vulkan.descriptor_set_allocator,
                                layout.clone(),
                                [
                                    WriteDescriptorSet::image_view_sampler(0, ray_color, self.sampler.clone()),
                                    WriteDescriptorSet::image_view_sampler(1, ray_albedo, self.sampler.clone()),
                                    WriteDescriptorSet::image_view_sampler(2, ray_normal, self.sampler.clone()),
                                    WriteDescriptorSet::image_view_sampler(3, ray_depth, self.sampler.clone()),
                                    WriteDescriptorSet::buffer(4, render_info),
                                ],
                            ).unwrap()
                        };

                        render_pass
                            .set_viewport(0, [self.viewport.clone()])
                            .bind_vertex_buffers(0, self.vertex_buffer.clone())
                            .bind_pipeline_graphics(self.denoiser_pipeline.clone())
                            .bind_descriptor_sets(
                                PipelineBindPoint::Graphics,
                                self.denoiser_pipeline.layout().clone(),
                                0,
                                denoiser_descriptor_set,
                            );

                        render_pass
                            .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
                            .unwrap();

                        self.imgui_renderer.draw_commands(
                            render_pass,
                            &self.vulkan,
                            imgui_draw_data,
                        );
                    }
                    let status = self.vulkan.end_frame(render_pass);
                    match status {
                        DrawStatus::Ok => (),
                        DrawStatus::ShouldRecreateSwapchain => {
                            self.recreate_swapchain = true;
                        }
                    }
                }
                event => {
                    self.imgui_platform.handle_event(self.imgui.io_mut(), &self.window, &event);
                }
            }
        });
    }

    pub fn add_circle(&mut self) -> &mut Circle {
        let index = self.scene.all_circles.len();
        self.scene.all_circles.push(Circle::new());
        let mut c = self.scene.all_circles.get_mut(index).unwrap();
        c.index = index;
        c.dirty;
        return c;
    }

    pub fn add_material(&mut self) -> &mut Material {
        let index = self.scene.all_materials.len();
        self.scene.all_materials.push(Material::new());
        let mut m = self.scene.all_materials.get_mut(index).unwrap();
        m.index = index;
        m.dirty;
        return m;
    }
}
