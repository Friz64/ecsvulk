use std::{
    sync::Arc,
    boxed::Box,
};
use objects::{
    Objects,
};
use config::{
    configloader::{Config},
};
use ecs::{
    components::*,
};
use graphics::pipeline::{
    PipelineSettings,
};
use ::winit::{
    WindowBuilder, EventsLoop, Window,
};
use ::vulkano_win::{
    self, VkSurfaceBuild, CreationError,
};
use ::vulkano::{
    instance::{self, debug::{DebugCallback, MessageTypes}, Instance, InstanceCreationError, ApplicationInfo, PhysicalDevice, Features},
    device::{Device, Queue, DeviceExtensions},
    swapchain::{self, Surface, CapabilitiesError, Swapchain, SwapchainCreationError, CompositeAlpha, PresentMode, AcquireError, ColorSpace},
    image::{swapchain::SwapchainImage, ImageUsage, attachment::AttachmentImage, sys::ImageCreationError, ImageDimensions},
    sync::{self, SharingMode, GpuFuture, FlushError},
    format::{Format, D16Unorm},
    framebuffer::{RenderPassAbstract, RenderPassCreationError, Subpass, Framebuffer, FramebufferAbstract, FramebufferCreationError},
    pipeline::{GraphicsPipeline, GraphicsPipelineAbstract, viewport::Viewport, vertex::TwoBuffersDefinition},
    buffer::{immutable::ImmutableBuffer, cpu_pool::CpuBufferPool, BufferAccess, BufferUsage, TypedBufferAccess},
    memory::{DeviceMemoryAllocError},
    descriptor::{descriptor_set::{FixedSizeDescriptorSetsPool, PersistentDescriptorSetBuildError}},
    command_buffer::{AutoCommandBufferBuilder, DynamicState, BeginRenderPassError, BuildError},
};
use ::cgmath::{
    self, Angle, InnerSpace,
};
use ::rayon::{
    ThreadPool,
};
use ::specs::{
    World, Join, Entity,
};
use ::graphics::{
    *,
};
use log::warn;

pub mod pipelines {
    use super::*;

    gen_pipelines!(
        [terrain, "shaders/terrain.vert", "shaders/terrain.frag", PipelineSettings {
            backfaceculling: true,
            wireframe: false,
        }],
        [normal, "shaders/normal.vert", "shaders/normal.frag", PipelineSettings {
            backfaceculling: true,
            wireframe: true,
        }]
    );
}

pub struct Renderer {
    pub focused: bool,
    pub cursor_grabbed: bool,

    #[allow(unused)]
    instance: Arc<Instance>,
    pub surface: Arc<Surface<Window>>,
    
    #[allow(unused)]
    physical_device: usize, // lifetime issues
    device: Arc<Device>,

    pub queue: Arc<Queue>,

    swap_chain: Arc<Swapchain<Window>>,
    swap_chain_images: Vec<Arc<SwapchainImage<Window>>>,

    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    pipelines: pipelines::Pipelines,

    depth_buffer: Arc<AttachmentImage<D16Unorm>>,
    swap_chain_framebuffers: Vec<Arc<FramebufferAbstract + Send + Sync>>,

    prev_frame: Option<Box<GpuFuture + Send + Sync>>,

    dynamic_state: DynamicState,

    recreate_swap_chain: bool,
}

impl Renderer {
    pub fn new(events_loop: &EventsLoop, config: &Config) -> (Self, Option<DebugCallback>) {
        let required_device_extensions = DeviceExtensions {
            khr_swapchain: true,
            .. DeviceExtensions::none()
        };

        let instance = Self::instance();
        let debug_callback = Self::callback(&instance);

        let surface = Self::surface(events_loop, &instance);

        let (physical_device, queue_family) = Self::physical_device(&instance, &surface, &required_device_extensions);
        let (device, queue) =  Self::logical_device(&instance, &required_device_extensions, physical_device, queue_family);

        let (swap_chain, swap_chain_images) = Self::swap_chain(&instance, &surface, physical_device, &device, &queue, config);

        let render_pass = Self::render_pass(&device, swap_chain.format());

        let pipelines = pipelines::Pipelines::new(&device, &render_pass);

        let depth_buffer = Self::depth_buffer(&device, swap_chain.dimensions())
            .unwrap_or_else(|| ::error_close!("Invalid dimensions"));
        let swap_chain_framebuffers = Self::framebuffers(&swap_chain_images, &render_pass, &depth_buffer);

        let dynamic_state = Self::dynamic_state(swap_chain.dimensions());

        let prev_frame = Some(Box::new(sync::now(device.clone())) as Box<_>);

        ::info!("Renderer initialized");

        (Renderer {
            focused: true,
            cursor_grabbed: false,

            instance,
            surface,

            physical_device,
            device,

            queue,

            swap_chain,
            swap_chain_images,

            render_pass,
            pipelines,

            depth_buffer,
            swap_chain_framebuffers,

            prev_frame,

            dynamic_state,

            recreate_swap_chain: false,
        }, debug_callback)
    }

    fn instance() -> Arc<Instance> {
        let layer = "VK_LAYER_LUNARG_standard_validation";

        let validation_supported = instance::layers_list()
            .unwrap_or_else(|err| ::error_close!("{}", err))
            .map(|layer| layer.name().to_owned())
            .any(|name| name == layer);

        let app_infos = ApplicationInfo {
            application_name: Some(::NAME.into()),
            application_version: Some(::VERSION),
            engine_name: Some((::NAME.to_string() + " Engine").into()),
            engine_version: Some(::VERSION),
        };

        let extensions = vulkano_win::required_extensions();

        // activates the validation layer if supported and if this is a debug build
        let instance = match ::DEBUG && validation_supported {
            true =>  instance::Instance::new(Some(&app_infos), &extensions, Some(layer)),
            false => instance::Instance::new(Some(&app_infos), &extensions, None)
        };

        instance.unwrap_or_else(|err| match err {
            InstanceCreationError::LoadingError(err) => ::error_close!("{}", err),
            InstanceCreationError::OomError(err) => ::error_close!("{}", err),
            _ => ::error_close!("{}", err),
        })
    }

    fn callback(instance: &Arc<Instance>) -> Option<DebugCallback> {
        let types = MessageTypes {
            error: true,
            warning: true,
            performance_warning: true,
            information: true,
            debug: true,
        };

        DebugCallback::new(&instance, types, |msg| {
            warn!("{} - {}", msg.layer_prefix, msg.description);
        }).ok()
    }

    fn surface(
        events_loop: &EventsLoop,
        instance: &Arc<Instance>
    ) -> Arc<Surface<Window>> {
        WindowBuilder::new()
            .with_title(::NAME.to_owned() + " " + &::VERSION.to_string())
            .build_vk_surface(&events_loop, instance.clone())
            .unwrap_or_else(|err| match err {
                    CreationError::SurfaceCreationError(err) => ::error_close!("{}", err),
                    CreationError::WindowCreationError(err) => ::error_close!("{}", err),
            })
    }

    fn physical_device(
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        required_extensions: &DeviceExtensions,
    ) -> (usize, u32) {
        let mut family = None;

        let device = PhysicalDevice::enumerate(&instance)
            .position(|device| {
                family = device.queue_families()
                    .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
                    .map(|val| Some(val))
                    .unwrap_or(None);
                
                let extensions_supported = DeviceExtensions::supported_by_device(device)
                    .intersection(required_extensions) == *required_extensions;
                
                let swap_chain_supported = if extensions_supported {
                    let capabilities = surface.capabilities(device)
                        .unwrap_or_else(|err| match err {
                            CapabilitiesError::OomError(err) => ::error_close!("{}", err),
                            _ => ::error_close!("{}", err),
                        });
                    
                    !capabilities.supported_formats.is_empty() &&       // at least one format supported
                    !capabilities.present_modes.iter().next().is_none() // at least one present mode supported
                } else { false };
                
                extensions_supported && swap_chain_supported
            })
            .unwrap_or_else(|| ::error_close!("No suitable GPU found"));
        let family = family
            .unwrap_or_else(|| ::error_close!("No suitable QueueFamily found"))
            .id();

        (device, family)
    }
    
    fn logical_device(
        instance: &Arc<Instance>,
        required_extensions: &DeviceExtensions,
        physical_device_index: usize,
        queue_family: u32,
    ) -> (Arc<Device>, Arc<Queue>) {
        let physical_device = PhysicalDevice::from_index(instance, physical_device_index)
            .unwrap_or_else(|| ::error_close!("Failed to reconstruct physical device from earlier obtained index"));

        let queue_family = (
            physical_device.queue_families().nth(queue_family as usize)
                .unwrap_or_else(|| ::error_close!("Failed to reconstruct queue family from earlier obtained index")),
            1.0,
        );

        let mut features = Features::none();
        features.fill_mode_non_solid = true;

        let (device, mut queues) = Device::new(physical_device, &features,
            required_extensions, [queue_family].iter().cloned())
            .unwrap_or_else(|err| ::error_close!("{}", err));

        let queue = queues.next()
            .unwrap_or_else(|| ::error_close!("No queues found"));

        (device, queue)
    }

    fn swap_chain(
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        physical_device_index: usize,
        device: &Arc<Device>,
        queue: &Arc<Queue>,
        config: &Config,
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        let physical_device = PhysicalDevice::from_index(instance, physical_device_index)
            .unwrap_or_else(|| ::error_close!("Failed to reconstruct physical device from earlier obtained index"));
        let capabilities = surface.capabilities(physical_device)
            .unwrap_or_else(|err| match err {
                CapabilitiesError::OomError(err) => ::error_close!("{}", err),
                _ => ::error_close!("{}", err),
            });

        let surface_format = {
            capabilities.supported_formats.iter()
            // we try to find our preferred format
            .find(|(format, color_space)|
                *format == Format::B8G8R8A8Unorm && *color_space == ColorSpace::SrgbNonLinear
            )
            // if that fails, we just use the first supported format
            .unwrap_or(&capabilities.supported_formats[0])
            //capabilities.supported_formats[0]
        };

        // https://docs.rs/vulkano/0.10.0/vulkano/swapchain/enum.PresentMode.html
        let present_mode = match config.graphics.settings.vsync { // fifo is always supported
            false if capabilities.present_modes.immediate => PresentMode::Immediate, // vsync not requested, capable of immediate
            true if !capabilities.present_modes.immediate => {
                warn!("VSync is disabled, but not syncing isn't available");
                PresentMode::Fifo
            }, // vsync requested, not capable of immediate
            _ => PresentMode::Fifo, // requested, use it
        };

        let dimensions = capabilities.current_extent.unwrap_or([800, 600]);

        let image_count = {
            let mut count = capabilities.min_image_count + 1;

            // if the max image count exists and is viable, use it instead
            if let Some(max_count) = capabilities.max_image_count {
                if count > max_count { count = max_count }
            }; count
        };

        let image_usage = ImageUsage {
            color_attachment: true,
            .. capabilities.supported_usage_flags
        };

        // https://docs.rs/vulkano/0.10.0/vulkano/sync/enum.SharingMode.html
        let sharing_mode: SharingMode = queue.into();

        let alpha = capabilities.supported_composite_alpha.iter()
            .next().unwrap_or(CompositeAlpha::Opaque);

        Swapchain::new( // https://docs.rs/vulkano/0.10.0/vulkano/swapchain/index.html
            // Create the swapchain in this `device`'s memory.
            device.clone(),
            // The surface where the images will be presented.
            surface.clone(),
            // How many buffers to use in the swapchain.
            image_count,
            // The format of the images.
            surface_format.0,
            // The size of each image.
            dimensions,
            // How many layers each image has.
            1,
            // What the images are going to be used for.
            image_usage,
            // Describes which queues will interact with the swapchain.
            sharing_mode,
            // What transformation to use with the surface.
            capabilities.current_transform,
            // How to handle the alpha channel.
            alpha,
            // How to present images.
            present_mode,
            // Clip the parts of the buffer which aren't visible.
            true,
            // Previous swapchain.
            None,
        ).unwrap_or_else(|err| match err {
            SwapchainCreationError::OomError(err) => ::error_close!("{}", err),
            _ => ::error_close!("{}", err),
        })
    }

    fn render_pass(
        device: &Arc<Device>,
        color_format: Format
    ) -> Arc<RenderPassAbstract + Send + Sync> {
        Arc::new(single_pass_renderpass!(device.clone(), // https://docs.rs/vulkano/0.10.0/vulkano/framebuffer/index.html
            attachments: {
                color: {
                    load: Clear,  // https://docs.rs/vulkano/0.10.0/vulkano/framebuffer/enum.LoadOp.html
                    store: Store, // https://docs.rs/vulkano/0.10.0/vulkano/framebuffer/enum.StoreOp.html
                    format: color_format,
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        ).unwrap_or_else(|err| match err {
            RenderPassCreationError::OomError(err) => ::error_close!("{}", err),
            _ => ::error_close!("{}", err),
        }))
    }

    fn depth_buffer(
        device: &Arc<Device>,
        dimensions: [u32; 2],
    ) -> Option<Arc<AttachmentImage<D16Unorm>>> {
        AttachmentImage::transient(
            device.clone(),
            dimensions,
            D16Unorm,
        )
        .map(|val| Some(val))
        .unwrap_or_else(|err| match err {
            ImageCreationError::AllocError(err) => match err {
                DeviceMemoryAllocError::OomError(err) => ::error_close!("{}", err),
                _ => ::error_close!("{}", err),
            },
            ImageCreationError::UnsupportedDimensions { dimensions } => match dimensions {
                ImageDimensions::Dim2d { .. } => None,
                _ => ::error_close!("{}", err),
            },
            _ => ::error_close!("{}", err),
        })
    }

    fn framebuffers(
        swap_chain_images: &Vec<Arc<SwapchainImage<Window>>>,
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
        depth_buffer: &Arc<AttachmentImage<D16Unorm>>
    ) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
        swap_chain_images.iter()
            .map(|image| { // https://docs.rs/vulkano/0.10.0/vulkano/framebuffer/struct.FramebufferBuilder.html
                let framebuffer: Arc<FramebufferAbstract + Send + Sync> = Arc::new(Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                        .unwrap_or_else(|err| match err {
                            FramebufferCreationError::OomError(err) => ::error_close!("{}", err),
                            _ => ::error_close!("{}", err),
                        })
                    .add(depth_buffer.clone())
                        .unwrap_or_else(|err| match err {
                            FramebufferCreationError::OomError(err) => ::error_close!("{}", err),
                            _ => ::error_close!("{}", err),
                        })
                    .build()
                    .unwrap_or_else(|err| match err {
                        FramebufferCreationError::OomError(err) => ::error_close!("{}", err),
                        FramebufferCreationError::AttachmentDimensionsIncompatible {expected, obtained} =>
                            ::error_close!("{}; expected {:?}, got {:?}", err, expected, obtained),
                        FramebufferCreationError::AttachmentsCountMismatch {expected, obtained} =>
                            ::error_close!("{}; expected {}, got {}", err, expected, obtained),
                        _ => ::error_close!("{}", err),
                    })
                ); framebuffer
            }
        ).collect::<Vec<_>>()
    }

    pub fn vertex_buffer(queue: &Arc<Queue>, vertices: &Vec<Vertex>) -> Arc<BufferAccess + Send + Sync> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            vertices.iter().cloned(), BufferUsage::vertex_buffer(), 
            queue.clone())
            .unwrap_or_else(|err| match err {
                DeviceMemoryAllocError::OomError(err) => ::error_close!("{}", err),
                _ => ::error_close!("{}", err),
            });
        future.flush().unwrap();
        buffer
    }

    pub fn normals_buffer(queue: &Arc<Queue>, normals: &Vec<Normal>) -> Arc<BufferAccess + Send + Sync> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            normals.iter().cloned(), BufferUsage::vertex_buffer(), 
            queue.clone())
            .unwrap_or_else(|err| match err {
                DeviceMemoryAllocError::OomError(err) => ::error_close!("{}", err),
                _ => ::error_close!("{}", err),
            });
        future.flush().unwrap();
        buffer
    }

    pub fn index_buffer(queue: &Arc<Queue>, indices: &Vec<u16>) -> Arc<TypedBufferAccess<Content=[u16]> + Send + Sync> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            indices.iter().cloned(), BufferUsage::index_buffer(), 
            queue.clone())
            .unwrap_or_else(|err| match err {
                DeviceMemoryAllocError::OomError(err) => ::error_close!("{}", err),
                _ => ::error_close!("{}", err),
            });
        future.flush().unwrap();
        buffer
    }

    fn dynamic_state(dimensions: [u32; 2]) -> DynamicState {
        DynamicState {
            line_width: None,
            viewports: Some(vec![
                Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                    depth_range: 0.0 .. 1.0,
                },
            ]),
            scissors: None,
        }
    }

    fn recreate_swap_chain(&mut self) -> bool {
        let physical_device = PhysicalDevice::from_index(&self.instance, self.physical_device)
            .unwrap_or_else(|| ::error_close!("Failed to reconstruct physical device from earlier obtained index"));

        let dimensions = self.surface.capabilities(physical_device)
            .unwrap_or_else(|err| match err {
                CapabilitiesError::OomError(err) => ::error_close!("{}", err),
                _ => ::error_close!("{}", err),
            }).current_extent.unwrap_or([800, 600]);

        let (swap_chain, swap_chain_images) = match self.swap_chain.recreate_with_dimension(dimensions) {
            Ok(r) => r,
            Err(SwapchainCreationError::OomError(err)) => ::error_close!("{}", err),
            Err(SwapchainCreationError::UnsupportedDimensions) => return true,
            Err(err) => ::error_close!("{}", err),
        };
        self.swap_chain = swap_chain;
        self.swap_chain_images = swap_chain_images;

        self.render_pass = Self::render_pass(&self.device, self.swap_chain.format());

        self.pipelines.recreate(&self.device, &self.render_pass);

        self.depth_buffer = match Self::depth_buffer(&self.device, dimensions) {
            Some(res) => res,
            None => return true,
        };
        self.swap_chain_framebuffers = Self::framebuffers(&self.swap_chain_images, &self.render_pass, &self.depth_buffer);

        self.dynamic_state.viewports = Self::dynamic_state(dimensions).viewports;

        self.recreate_swap_chain = false;

        false
    }

    fn update_player(delta_time: &f32, ecs: &World, player: &Entity, config: &Config) -> (Point3, Vec3) {
        let mut pos_storage = ecs.write_storage::<Pos>();
        let pyr_storage = ecs.read_storage::<PitchYawRoll>();
        let mut speed_storage = ecs.write_storage::<SpeedMultiplier>();

        let player_pos = pos_storage.get_mut(*player)
            .unwrap_or_else(|| ::error_close!("Invalid Player Entity; missing Pos Component"));
        let player_pyr = pyr_storage.get(*player)
            .unwrap_or_else(|| ::error_close!("Invalid Player Entity; missing PitchYawRoll Component"));
        let player_speed = speed_storage.get_mut(*player)
            .unwrap_or_else(|| ::error_close!("Invalid Player Entity; missing SpeedMultiplier Component"));

        let front = Vec3::new(
            cgmath::Rad(player_pyr.1).cos() * cgmath::Rad(player_pyr.0).cos(),
            cgmath::Rad(player_pyr.0).sin(),
            cgmath::Rad(player_pyr.1).sin() * cgmath::Rad(player_pyr.0).cos(),
        );

        let scrollspeed = config.controls.sensitivity.scroll_speed * 0.2;

        // update player speed multiplier
        if config.controls.movement.speed_up.down() {
            player_speed.0 += scrollspeed;
        } else if config.controls.movement.speed_down.down() {
            player_speed.0 -= scrollspeed;
        }

        if player_speed.0 <= 0.0 {
            player_speed.0 = 0.1;
        }

        // update player pos
        let delta_movement = config.controls.sensitivity.movement_speed * delta_time * 1.2 * player_speed.0;
        let frontcross = front.cross(Vec3::new(0.0, 1.0, 0.0)).normalize();

        if config.controls.movement.forwards.down_hold() {
            player_pos.0 -= delta_movement * Vec3::new(-frontcross.z, frontcross.y, frontcross.x);
        }
        if config.controls.movement.backwards.down_hold() {
            player_pos.0 += delta_movement * Vec3::new(-frontcross.z, frontcross.y, frontcross.x);
        }
        if config.controls.movement.left.down_hold() {
            player_pos.0 -= delta_movement * frontcross;
        }
        if config.controls.movement.right.down_hold() {
            player_pos.0 += delta_movement * frontcross;
        }
        if config.controls.movement.down.down_hold() {
            player_pos.0.y -= delta_movement;
        }
        if config.controls.movement.up.down_hold() {
            player_pos.0.y += delta_movement;
        }

        let pos = Point3::new(player_pos.0.x, player_pos.0.y, player_pos.0.z);

        (pos, front)
    }

    fn pv_matrices(dimensions: [u32; 2], pos: Point3, front: Vec3) -> (Mat4, Mat4) {
        let mut proj = cgmath::perspective( // projection
            cgmath::Rad(::std::f32::consts::FRAC_PI_2),  // fov
            dimensions[0] as f32 / dimensions[1] as f32, // aspect ratio
            0.01, 200.0,                                 // near and far plane
        );

        // the vulkan coordinate system is in australia
        proj.y.y *= -1.0;

        let view = Mat4::look_at_dir(   // view
            pos,                    // position
            front,                  // pitch yaw
            (0.0, 1.0, 0.0).into(), // up
        );

        (proj, view)
    }

    fn model_matrix(translation: Vec3, pitch: f32, yaw: f32, roll: f32) -> Mat4 {
        Mat4::from_translation(translation) *
        Mat4::from_angle_x(cgmath::Deg(pitch)) *
        Mat4::from_angle_y(cgmath::Deg(yaw)) *
        Mat4::from_angle_z(cgmath::Deg(roll))
    }

    pub fn draw(&mut self, delta_time: &f32, ecs: &World, player: &Entity, objects: &Objects, config: &Config) -> bool {
        if {
            // windows resizes window to 0, 0 upon minimizing
            let dimensions: (u32, u32) = self.surface.window().get_inner_size()
                .unwrap_or_else(|| ::error_close!("Failed to get the current window dimensions")).into();
            if dimensions == (0, 0) {
                return true
            };

            if self.recreate_swap_chain {
                if self.recreate_swap_chain() { return true }
            };

            false
        } { return true; };

        let (image_index, acquire_future) = match {
            match swapchain::acquire_next_image(self.swap_chain.clone(), None) {
                Ok(r) => Some(r),
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swap_chain = true;
                    None
                },
                Err(err) => ::error_close!("{}", err),
            }
        } {
            Some(r) => r,
            None => return true,
        };

        self.prev_frame.as_mut()
            .unwrap_or_else(|| ::error_close!("Prev frame is None (this is impossible)"))
            .cleanup_finished();

        // update engine config stuff
        if config.controls.engine.wireframe.down() {
            let mut wireframe_storage = ecs.write_storage::<Wireframe>();
            let player_wireframe = wireframe_storage.get_mut(*player)
                .unwrap_or_else(|| ::error_close!("Invalid Player Entity; missing Wireframe Component"));
                
            player_wireframe.0 = !player_wireframe.0;

            let mut settings = self.pipelines.terrain.settings.clone();
            settings.wireframe = player_wireframe.0;

            self.pipelines.terrain = pipelines::terrain::Pipeline::new(&self.device, &self.render_pass, settings);
        }

        let (projection, view) =  {
            let (pos, front) = Self::update_player(delta_time, ecs, player, config);

            Self::pv_matrices(self.swap_chain.dimensions(), pos, front)
        };

        let command_buffer = {
            let mut command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family())
                .unwrap_or_else(|err| ::error_close!("{}", err))
                .begin_render_pass(
                    self.swap_chain_framebuffers[image_index].clone(), // requested swapchain framebuffer
                    false, // no secondary command buffer
                    vec![ // clear values
                        [0.3, 0.3, 0.3, 1.0].into(), // color
                        1f32.into()                  // depth
                    ],
                )
                .unwrap_or_else(|err| match err {
                    BeginRenderPassError::AutoCommandBufferBuilderContextError(err) => ::error_close!("{}", err),
                    BeginRenderPassError::SyncCommandBufferBuilderError(err) => ::error_close!("{}", err),
                });

            let pipelines = ecs.read_storage::<Pipeline>();
            let positions = ecs.read_storage::<Pos>();
            let pitchyawroll = ecs.read_storage::<PitchYawRoll>();
            let models = ecs.read_storage::<Model>();

            for (pos, PitchYawRoll(pitch, yaw, roll), model, pipeline) in (&positions, &pitchyawroll, &models, &pipelines).join() {
                match pipeline.0 {
                    pipelines::Pipeline::terrain => {
                        let model_mat = Self::model_matrix(pos.0, *pitch, *yaw, *roll);
                        let main_set = {
                            Arc::new(self.pipelines.terrain.sets_pool.next()
                                .add_buffer(self.pipelines.terrain.cbp.next(pipelines::terrain::vs::ty::Data {
                                        mvp: (projection * view * model_mat).into(),
                                        //view: view.into(),
                                        //model: model_mat.into(),
                                    }).unwrap_or_else(|err| match err {
                                        DeviceMemoryAllocError::OomError(err) => ::error_close!("{}", err),
                                        _ => ::error_close!("{}", err),
                                    })).unwrap_or_else(|err| ::error_close!("{}", err))
                                .build().unwrap_or_else(|err| match err {
                                    PersistentDescriptorSetBuildError::OomError(err) => ::error_close!("{}", err),
                                    _ => ::error_close!("{}", err),
                                })
                            )
                        };

                        if let Some((vertex_buf, normals_buf, index_buf)) = model.0.get_buffers(objects) {
                            command_buffer = command_buffer.draw_indexed(
                                self.pipelines.terrain.pipeline.clone(),
                                &self.dynamic_state,
                                [vertex_buf.clone(), normals_buf.clone()].to_vec(),
                                index_buf.clone(),
                                main_set.clone(), ()
                            ).unwrap();
                        }
                    },
                    pipelines::Pipeline::normal => {
                        let model_mat = Self::model_matrix(pos.0, *pitch, *yaw, *roll);
                        let main_set = {
                            Arc::new(self.pipelines.normal.sets_pool.next()
                                .add_buffer(self.pipelines.normal.cbp.next(pipelines::normal::vs::ty::Data {
                                        mvp: (projection * view * model_mat).into(),
                                        //view: view.into(),
                                        //model: model_mat.into(),
                                    }).unwrap_or_else(|err| match err {
                                        DeviceMemoryAllocError::OomError(err) => ::error_close!("{}", err),
                                        _ => ::error_close!("{}", err),
                                    })).unwrap_or_else(|err| ::error_close!("{}", err))
                                .build().unwrap_or_else(|err| match err {
                                    PersistentDescriptorSetBuildError::OomError(err) => ::error_close!("{}", err),
                                    _ => ::error_close!("{}", err),
                                })
                            )
                        };

                        if let Some((vertex_buf, normals_buf, index_buf)) = model.0.get_buffers(objects) {
                            command_buffer = command_buffer.draw_indexed(
                                self.pipelines.normal.pipeline.clone(),
                                &self.dynamic_state,
                                [vertex_buf.clone(), normals_buf.clone()].to_vec(),
                                index_buf.clone(),
                                main_set.clone(), ()
                            ).unwrap();
                        }
                    }
                };
            }

            command_buffer.end_render_pass()
                .unwrap_or_else(|err| ::error_close!("{}", err))
                .build()
                .unwrap_or_else(|err| match err {
                    BuildError::AutoCommandBufferBuilderContextError(err) => ::error_close!("{}", err),
                    BuildError::OomError(err) => ::error_close!("{}", err),
                })
        };

        {
            match self.prev_frame.take().unwrap_or(Box::new(sync::now(self.device.clone())) as Box<_>)
                .join(acquire_future)
                .then_execute(self.queue.clone(), command_buffer)
                    .unwrap_or_else(|err| ::error_close!("{}", err))
                .then_swapchain_present(self.queue.clone(), self.swap_chain.clone(), image_index)
                .then_signal_fence_and_flush()
            {
                Ok(future) => {
                    self.prev_frame = Some(Box::new(future) as Box<_>);
                },
                Err(FlushError::OutOfDate) => {
                    self.recreate_swap_chain = true;
                    self.prev_frame = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
                },
                Err(err) => {
                    warn!("{}", err);
                    self.prev_frame = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
                },
            };
        };

        // drawing didn't fail
        false
    }
}