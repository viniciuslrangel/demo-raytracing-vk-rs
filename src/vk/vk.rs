use std::cell::RefCell;
use std::cmp::max;
use std::sync::Arc;
use std::time::Duration;

use vulkano::{sync, Version, VulkanLibrary};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassContents};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::format::Format;
use vulkano::image::{AttachmentImage, ImageAccess, ImageUsage, SwapchainImage};
use vulkano::image::view::ImageView;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::instance::debug::{DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger, DebugUtilsMessengerCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::VertexDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::EntryPoint;
use vulkano::swapchain::{acquire_next_image, AcquireError, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainCreationError, SwapchainPresentInfo};
use vulkano::sync::{FlushError, GpuFuture};
use winit::window::Window;

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

pub struct Buffers {
    pub raytrace_fb: Arc<Framebuffer>,
    pub screen_fb: Arc<Framebuffer>,

    pub ray_color_image: Arc<ImageView<AttachmentImage>>,
    pub ray_albedo_image: Arc<ImageView<AttachmentImage>>,
    pub ray_normal_image: Arc<ImageView<AttachmentImage>>,
    pub ray_depth_image: Arc<ImageView<AttachmentImage>>,
}

pub struct Vk {
    pub device_name: String,

    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub surface: Arc<Surface>,
    pub swapchain: Arc<Swapchain>,
    pub images: Vec<Arc<SwapchainImage>>,

    pub uploads: Option<RefCell<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>>,
    pub raytrace_render_pass: Arc<RenderPass>,
    pub screen_render_pass: Arc<RenderPass>,

    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub command_buffer_allocator: StandardCommandBufferAllocator,

    pub uniform_buffer: SubbufferAllocator,
    pub storage_buffer: SubbufferAllocator,
    pub buffers: Option<Vec<Buffers>>,

    pub previous_frame_end: Option<Box<dyn GpuFuture>>,
    should_recreate_swapchain: bool,
    acquire_future: Option<SwapchainAcquireFuture>,
    current_image_index: u32,
}

pub enum DrawStatus {
    Ok,
    ShouldRecreateSwapchain,
}

impl<'a> Vk {
    pub fn create_instance() -> Arc<Instance> {
        let library = VulkanLibrary::new()
            .expect("Failed to load Vulkan library");

        let mut required_extensions = vulkano_win::required_extensions(&library);
        required_extensions.ext_debug_utils = ENABLE_VALIDATION_LAYERS;

        if ENABLE_VALIDATION_LAYERS {
            println!("List of Vulkan debugging layers available to use:");
            let layers = library.layer_properties().unwrap();
            for l in layers {
                println!("\t{}", l.name());
            }
        }

        let mut info = InstanceCreateInfo {
            enabled_extensions: required_extensions,
            enumerate_portability: true,
            application_name: Some("Raytracing Demo".to_string()),
            application_version: Version::major_minor(1, 0),
            ..Default::default()
        };

        if ENABLE_VALIDATION_LAYERS {
            let layers = vec!["VK_LAYER_KHRONOS_validation".to_owned()];
            info.enabled_layers = layers;
        }

        Instance::new(library, info)
            .expect("Failed to create Vulkan instance")
    }

    pub fn create_device(instance: Arc<Instance>, surface: Arc<Surface>) -> Self {
        if ENABLE_VALIDATION_LAYERS {
            let messenger = setup_debug_callback(&instance);
            Box::leak(Box::new(messenger));
        }

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .map(|p| {
                (!p.queue_family_properties().is_empty())
                    .then_some((p, 0))
                    .expect("couldn't find a queue family")
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("no device available");

        let physical_properties = physical_device.properties();
        println!(
            "Using device: {} (type: {:?})",
            physical_properties.device_name,
            physical_properties.device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        ).expect("failed to create device");
        let queue = queues.next().unwrap();

        let (swapchain, images) = {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            let image_format = Some(
                device
                    .physical_device()
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0]
                    .0,
            );
            let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: max(4, surface_capabilities.min_image_count),
                    image_format,
                    image_extent: window.inner_size().into(),

                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),

                    ..Default::default()
                },
            ).unwrap()
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(device.clone()));

        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        );

        let uniform_buffer = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        );

        let storage_buffer = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
        );

        let previous_frame_end = Some(sync::now(device.clone()).boxed());

        let uploads = RefCell::new(AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap());

        let raytrace_render_pass = vulkano::ordered_passes_renderpass!(
            device.clone(),
            attachments: {
                raytracing_output: {
                    load: DontCare,
                    store: Store,
                    format: Format::R32G32B32A32_SFLOAT,
                    samples: 1,
                },
                raytracing_albedo: {
                    load: DontCare,
                    store: Store,
                    format: Format::B8G8R8A8_SRGB,
                    samples: 1,
                },
                raytracing_normal: {
                    load: DontCare,
                    store: Store,
                    format: Format::B8G8R8A8_SRGB,
                    samples: 1,
                },
                raytracing_depth: {
                    load: DontCare,
                    store: Store,
                    format: Format::R32_SFLOAT,
                    samples: 1,
                },
            },
            passes: [
                {
                    color: [raytracing_output, raytracing_albedo, raytracing_normal, raytracing_depth],
                    depth_stencil: {},
                    input: [],
                },
            ],
        ).unwrap();

        let screen_render_pass = vulkano::ordered_passes_renderpass!(
            device.clone(),
            attachments: {
                screen_output: {
                    load: Clear,
                    store: Store,
                    format: swapchain.image_format(),
                    samples: 1,
                },
                raytracing_output: {
                    load: Load,
                    store: DontCare,
                    format: Format::R32G32B32A32_SFLOAT,
                    samples: 1,
                },
                raytracing_albedo: {
                    load: Load,
                    store: DontCare,
                    format: Format::B8G8R8A8_SRGB,
                    samples: 1,
                },
                raytracing_normal: {
                    load: Load,
                    store: DontCare,
                    format: Format::B8G8R8A8_SRGB,
                    samples: 1,
                },
                raytracing_depth: {
                    load: Load,
                    store: DontCare,
                    format: Format::R32_SFLOAT,
                    samples: 1,
                },
            },
            passes: [
                {
                    color: [screen_output],
                    depth_stencil: {},
                    input: [raytracing_output, raytracing_albedo, raytracing_normal, raytracing_depth],
                },
            ],
        ).unwrap();

        return Vk {
            device_name: physical_properties.device_name.clone(),

            instance,
            device,
            queue,
            surface,
            swapchain,
            images,

            uploads: Some(uploads),
            raytrace_render_pass,
            screen_render_pass,

            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,

            uniform_buffer,
            storage_buffer,
            buffers: None,

            previous_frame_end,
            should_recreate_swapchain: false,
            acquire_future: None,
            current_image_index: 0,
        };
    }

    pub fn create_pipeline<T>(
        &self,
        subpass: Subpass,
        vertex_input_state: T,
        vertex_shader: EntryPoint,
        fragment_shader: EntryPoint,
        blend_count: u32,
    ) -> Arc<GraphicsPipeline>
        where T: VertexDefinition
    {
        GraphicsPipeline::start()
            .render_pass(subpass)
            .vertex_input_state(vertex_input_state)
            .input_assembly_state(InputAssemblyState::new())
            .vertex_shader(vertex_shader, ())
            .fragment_shader(fragment_shader, ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .color_blend_state(ColorBlendState::new(blend_count))
            .build(self.device.clone())
            .unwrap()
    }

    pub fn setup_framebuffer(&mut self, viewport: &mut Viewport) {
        let dimensions = self.images[0].dimensions().width_height();
        viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

        let count = self.images.len();

        let buffers = (0..count).map(|idx| {
            let ray_color_image = ImageView::new_default(
                AttachmentImage::with_usage(
                    &self.memory_allocator,
                    dimensions,
                    Format::R32G32B32A32_SFLOAT,
                    ImageUsage::INPUT_ATTACHMENT | ImageUsage::SAMPLED,
                ).unwrap(),
            ).unwrap();

            let ray_albedo_image = ImageView::new_default(
                AttachmentImage::with_usage(
                    &self.memory_allocator,
                    dimensions,
                    Format::B8G8R8A8_SRGB,
                    ImageUsage::INPUT_ATTACHMENT | ImageUsage::SAMPLED,
                ).unwrap(),
            ).unwrap();

            let ray_normal_image = ImageView::new_default(
                AttachmentImage::with_usage(
                    &self.memory_allocator,
                    dimensions,
                    Format::B8G8R8A8_SRGB,
                    ImageUsage::INPUT_ATTACHMENT | ImageUsage::SAMPLED,
                ).unwrap(),
            ).unwrap();

            let ray_depth_image = ImageView::new_default(
                AttachmentImage::with_usage(
                    &self.memory_allocator,
                    dimensions,
                    Format::R32_SFLOAT,
                    ImageUsage::INPUT_ATTACHMENT | ImageUsage::SAMPLED,
                ).unwrap(),
            ).unwrap();

            let screen_output = ImageView::new_default(self.images[idx].clone()).unwrap();

            let raytrace_fb = Framebuffer::new(
                self.raytrace_render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![
                        ray_color_image.clone(),
                        ray_albedo_image.clone(),
                        ray_normal_image.clone(),
                        ray_depth_image.clone(),
                    ],
                    ..Default::default()
                },
            ).unwrap();

            let screen_fb = Framebuffer::new(
                self.screen_render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![
                        screen_output,
                        ray_color_image.clone(),
                        ray_albedo_image.clone(),
                        ray_normal_image.clone(),
                        ray_depth_image.clone(),
                    ],
                    ..Default::default()
                },
            ).unwrap();

            Buffers {
                raytrace_fb,
                screen_fb,
                ray_color_image,
                ray_albedo_image,
                ray_normal_image,
                ray_depth_image,
            }
        }).collect();

        self.buffers = Some(buffers);
    }

    pub fn recreate_swapchain(&mut self, size: [u32; 2], viewport: &mut Viewport) {
        let (new_swapchain, new_images) =
            match self.swapchain.recreate(SwapchainCreateInfo {
                image_extent: size,
                ..self.swapchain.create_info()
            }) {
                Ok(r) => r,
                Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                Err(err) => panic!("failed to recreate swapchain: {}", err),
            };

        self.swapchain = new_swapchain;
        self.images = new_images;

        self.setup_framebuffer(viewport);
    }

    pub fn wait_frame(&mut self) {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();
    }

    pub fn do_upload(&mut self) {
        self.wait_frame();
        let uploads = self.uploads.take().unwrap();
        let uploads = uploads.into_inner();
        self.previous_frame_end = Some(
            uploads
                .build()
                .unwrap()
                .execute(self.queue.clone())
                .unwrap()
                .boxed(),
        );
    }

    pub fn begin_frame(&mut self) -> Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>> {
        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), Some(Duration::from_secs(1))) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.should_recreate_swapchain = true;
                    return None;
                }
                Err(e) => {
                    println!("failed to acquire next image: {e}");
                    return None;
                }
            };
        self.should_recreate_swapchain = suboptimal;
        self.acquire_future = Some(acquire_future);
        self.current_image_index = image_index;

        let queue_index = self.queue.queue_family_index();
        let mut command_builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            queue_index,
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        command_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: (0..self.raytrace_render_pass.attachments().len())
                        .map(|_| Some([0.0, 0.0, 1.0, 1.0].into()))
                        .collect(),
                    ..RenderPassBeginInfo::framebuffer(
                        self.buffers.as_ref().unwrap()[image_index as usize].raytrace_fb.clone(),
                    )
                },
                SubpassContents::Inline,
            )
            .unwrap();

        return Some(command_builder);
    }

    pub fn next_render_pass(&mut self, command_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) -> &Buffers {
        let mut first_cmd_builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        std::mem::swap(command_builder, &mut first_cmd_builder);

        first_cmd_builder
            .end_render_pass()
            .unwrap();
        let raytrace_cmd = first_cmd_builder
            .build()
            .unwrap();

        let future = self.previous_frame_end
            .take()
            .unwrap()
            .then_execute(self.queue.clone(), raytrace_cmd)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .boxed();
        self.previous_frame_end = Some(future);

        let buf = &self.buffers.as_ref().unwrap()[self.current_image_index as usize];
        command_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: (0..self.screen_render_pass.attachments().len())
                        .map(|_| Some([0.0, 1.0, 0.0, 1.0].into()))
                        .collect(),
                    ..RenderPassBeginInfo::framebuffer(
                        buf.screen_fb.clone(),
                    )
                },
                SubpassContents::Inline,
            )
            .unwrap();

        return buf;
    }

    pub fn end_frame(&mut self, command_builder: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>) -> DrawStatus {
        if command_builder.is_none() {
            return match self.should_recreate_swapchain {
                true => {
                    self.should_recreate_swapchain = false;
                    DrawStatus::ShouldRecreateSwapchain
                }
                _ => DrawStatus::Ok
            };
        }

        let mut command_builder = command_builder.unwrap();

        command_builder
            .end_render_pass()
            .unwrap();

        let command_buffer = command_builder.build().unwrap();

        let future = self.previous_frame_end
            .take()
            .unwrap()
            .join(self.acquire_future.take().expect("start_frame() must be called before end_frame()"))
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), self.current_image_index),
            )
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
                self.should_recreate_swapchain = true;
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                println!("failed to flush future: {:?}", e);
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }

        if self.should_recreate_swapchain {
            self.should_recreate_swapchain = false;
            return DrawStatus::ShouldRecreateSwapchain;
        }
        return DrawStatus::Ok;
    }
}

fn setup_debug_callback(instance: &Arc<Instance>) -> DebugUtilsMessenger {
    return unsafe {
        DebugUtilsMessenger::new(
            instance.clone(),
            DebugUtilsMessengerCreateInfo {
                message_severity: DebugUtilsMessageSeverity::ERROR
                    | DebugUtilsMessageSeverity::WARNING
                    | DebugUtilsMessageSeverity::INFO
                    | DebugUtilsMessageSeverity::VERBOSE,
                message_type: DebugUtilsMessageType::GENERAL
                    | DebugUtilsMessageType::VALIDATION
                    | DebugUtilsMessageType::PERFORMANCE,
                ..DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|msg| {
                    let severity = if msg.severity.intersects(DebugUtilsMessageSeverity::ERROR) {
                        "error"
                    } else if msg.severity.intersects(DebugUtilsMessageSeverity::WARNING) {
                        "warning"
                    } else if msg.severity.intersects(DebugUtilsMessageSeverity::INFO) {
                        "information"
                    } else if msg.severity.intersects(DebugUtilsMessageSeverity::VERBOSE) {
                        "verbose"
                    } else {
                        panic!("no-impl");
                    };

                    let ty = if msg.ty.intersects(DebugUtilsMessageType::GENERAL) {
                        "general"
                    } else if msg.ty.intersects(DebugUtilsMessageType::VALIDATION) {
                        "validation"
                    } else if msg.ty.intersects(DebugUtilsMessageType::PERFORMANCE) {
                        "performance"
                    } else {
                        panic!("no-impl");
                    };

                    println!(
                        "{} {} {}: {}",
                        msg.layer_prefix.unwrap_or("unknown"),
                        ty,
                        severity,
                        msg.description
                    );
                }))
            },
        ).expect("Failed to create debug callback")
    };
}
