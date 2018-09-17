use std::{
    sync::Arc,
    boxed::Box,
};
use logger::{
    Logger, LogType,
};
use ::winit::{
    WindowBuilder,
    EventsLoop,
    Window,
};
use ::vulkano_win::{
    self,
    VkSurfaceBuild,
    CreationError,
};
use ::vulkano::{
    instance::{self, debug::{DebugCallback, MessageTypes}, Instance, InstanceCreationError, ApplicationInfo, PhysicalDevice, Features},
    device::{Device, Queue, DeviceExtensions},
    swapchain::{self, Surface, CapabilitiesError, Swapchain, SwapchainCreationError, CompositeAlpha, ColorSpace, PresentMode, AcquireError, SurfaceTransform},
    image::{swapchain::SwapchainImage, ImageUsage, attachment::AttachmentImage, sys::ImageCreationError},
    sync::{self, SharingMode, GpuFuture},
    format::{Format, D16Unorm},
    framebuffer::{RenderPassAbstract, RenderPassCreationError, Subpass, Framebuffer, FramebufferAbstract, FramebufferCreationError},
    pipeline::{GraphicsPipeline, GraphicsPipelineAbstract, viewport::Viewport},
    buffer::{cpu_access::CpuAccessibleBuffer, cpu_pool::CpuBufferPool, BufferAccess, BufferUsage},
    memory::{DeviceMemoryAllocError},
    descriptor::{descriptor_set::{PersistentDescriptorSet, PersistentDescriptorSetBuildError}},
    command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState, BeginRenderPassError},
};
use ::cgmath::{
    self, Angle,
};

pub type Vec3 = cgmath::Vector3<f32>;
pub type Point3 = cgmath::Point3<f32>;
pub type Mat4 = cgmath::Matrix4<f32>;

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
}

impl Vertex {
    pub fn new(pos: Vec3, normal: Vec3) -> Self {
        Self {
            pos:    pos.into(),
            normal: normal.into(),
        }
    }
}
impl_vertex!(Vertex, pos, normal);

// i think this goes over every queue family (thing that contains commands), checks if the queue family contains the wanted feature/command and saves its index
#[derive(Debug)]
struct QueueFamilyIndices { // plural index => indices
    graphics: Option<u16>,
    present: Option<u16>,
}

impl QueueFamilyIndices {
    fn new() -> Self {
        Self { graphics: None, present: None }
    }

    // checks if both indices have been found
    fn is_complete(&self) -> bool {
        self.graphics.is_some() && self.present.is_some()
    }
}

fn required_queue_families(
    logger: &mut Logger, 
    surface: &Arc<Surface<Window>>, 
    device: &PhysicalDevice
) -> QueueFamilyIndices {
    let mut indices = QueueFamilyIndices::new();

    for (i, queue_family) in device.queue_families().enumerate() {
        // queue family can do graphics
        let graphics_support = queue_family.supports_graphics();

        // queue family can draw on "surface"
        let present_support = surface.is_supported(queue_family)
            .unwrap_or_else(|err| match err {
                CapabilitiesError::OomError(err) => logger.error("SurfaceSupported", err),
                _ => logger.error("SurfaceSupported", err),
            });

        // save the index into the struct
        if graphics_support { indices.graphics = Some(i as u16) }
        if present_support { indices.present = Some(i as u16) }
        
        // indices have been found, we are done here
        if indices.is_complete() { break }
    }

    indices
}

// the cleanest way i found of implementing this
mod pipelines {
    use super::*;

    gen_pipeline!(main, "shaders/main.vert", "shaders/main.frag");
}

pub struct Renderer {
    pub focused: bool,

    instance: Arc<Instance>,
    surface: Arc<Surface<Window>>,

    physical_device: usize, // lifetime issues
    device: Arc<Device>,

    graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,

    swap_chain: Arc<Swapchain<Window>>,
    swap_chain_images: Vec<Arc<SwapchainImage<Window>>>,

    render_pass: Arc<RenderPassAbstract + Send + Sync>,

    main_pipeline: pipelines::main::Pipeline,

    depth_buffer: Arc<AttachmentImage<D16Unorm>>,
    swap_chain_framebuffers: Vec<Arc<FramebufferAbstract + Send + Sync>>,

    vertex_buffer: Arc<BufferAccess + Send + Sync>,
    normals_buffer: Arc<BufferAccess + Send + Sync>,
    index_buffer: Arc<BufferAccess + Send + Sync>,

    prev_frame_end: Option<Box<GpuFuture + Send + Sync>>,
    recreate_swap_chain: bool,
}

impl Renderer {
    pub fn new(logger: &mut Logger, events_loop: &EventsLoop) -> (Self, Option<DebugCallback>) {
        let required_device_extensions = DeviceExtensions {
            khr_swapchain: true,
            .. DeviceExtensions::none()
        };

        let instance = Self::instance(logger);
        let debug_callback = Self::callback(&instance);

        let surface = Self::surface(logger, events_loop, &instance);

        let physical_device = Self::physical_device(logger, &instance, &surface, &required_device_extensions);
        let (device, graphics_queue, present_queue) = Self::logical_device(logger, &instance, &surface, &required_device_extensions, physical_device);

        let (swap_chain, swap_chain_images) = Self::swap_chain(logger, &instance, &surface, physical_device, &device, &graphics_queue, &present_queue);

        let render_pass = Self::render_pass(logger, &device, swap_chain.format());
        let main_pipeline = Self::pipelines(logger, &device, swap_chain.dimensions(), &render_pass);

        let depth_buffer = Self::depth_buffer(logger, &device, swap_chain.dimensions());
        let swap_chain_framebuffers = Self::framebuffers(logger, &swap_chain_images, &render_pass, &depth_buffer);

        let vertex_buffer  = Self::vertex_buffer(logger, &device, vec![]);
        let normals_buffer = Self::normals_buffer(logger, &device, vec![]);
        let index_buffer   = Self::index_buffer(logger, &device, vec![]);

        let prev_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);

        logger.info("Renderer", "Renderer initialized");

        (Renderer {
            focused: true,

            instance,
            surface,

            physical_device,
            device,

            graphics_queue,
            present_queue,

            swap_chain,
            swap_chain_images,

            render_pass,

            main_pipeline,

            depth_buffer,
            swap_chain_framebuffers,

            vertex_buffer,
            normals_buffer,
            index_buffer,

            prev_frame_end,
            recreate_swap_chain: false,
        }, debug_callback)
    }

    fn instance(logger: &mut Logger) -> Arc<Instance> {
        let layer = "VK_LAYER_LUNARG_standard_validation";

        let validation_supported = instance::layers_list()
            .unwrap_or_else(|err| logger.error("GetLayersList", err))
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
            InstanceCreationError::LoadingError(err) => logger.error("InstanceCreate", err),
            InstanceCreationError::OomError(err) => logger.error("InstanceCreate", err),
            _ => logger.error("InstanceCreate", err),
        })
    }

    fn callback(instance: &Arc<Instance>) -> Option<DebugCallback> {
        let types = MessageTypes {
            error: true,
            warning: true,
            performance_warning: true,
            information: false,
            debug: true,
        };

        DebugCallback::new(&instance, types, |msg| {
            println!("{}", LogType::WARNING.gen_msg(msg.layer_prefix, msg.description));
        }).ok()
    }

    fn surface(
        logger: &mut Logger,
        events_loop: &EventsLoop,
        instance: &Arc<Instance>
    ) -> Arc<Surface<Window>> {
        WindowBuilder::new()
            .with_title(::NAME)
            .build_vk_surface(&events_loop, instance.clone())
            .unwrap_or_else(|err| match err {
                    CreationError::SurfaceCreationError(err) => logger.error("WindowCreate", err),
                    CreationError::WindowCreationError(err) => logger.error("WindowCreate", err),
            })
    }

    fn physical_device(
        logger: &mut Logger,
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        required_extensions: &DeviceExtensions,
    ) -> usize {
        PhysicalDevice::enumerate(&instance)
            .position(|device| {
                // check if the device is suitable
                let queue_families_supported = required_queue_families(logger, surface, &device)
                    .is_complete();

                let extensions_supported = DeviceExtensions::supported_by_device(device)
                    .intersection(required_extensions) == *required_extensions;
                
                let swap_chain_supported = if extensions_supported {
                    let capabilities = surface.capabilities(device)
                        .unwrap_or_else(|err| match err {
                            CapabilitiesError::OomError(err) => logger.error("SurfaceCapabilities", err),
                            _ => logger.error("SurfaceCapabilities", err),
                        });
                    
                    !capabilities.supported_formats.is_empty() &&       // at least one format supported
                    !capabilities.present_modes.iter().next().is_none() // at least one present mode supported
                } else { false };
                
                queue_families_supported && extensions_supported && swap_chain_supported
            })
            .unwrap_or_else(|| logger.error("FindGPU", "No suitable GPU found"))
    }
    
    fn logical_device(
        logger: &mut Logger,
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        required_extensions: &DeviceExtensions,
        physical_device_index: usize,
    ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        let physical_device = PhysicalDevice::from_index(instance, physical_device_index)
            .unwrap_or_else(|| logger.error("ReconstructPhysical", "Failed to reconstruct physical device from earlier obtained index"));
        let indices = required_queue_families(logger, surface, &physical_device);

        let queue_priority = 1.0; // only matters when using multiple queues
        let queue_families = [
            indices.graphics.map(|x| x as i32).unwrap_or(-1),
            indices.present.map(|x| x as i32).unwrap_or(-1),
        ]   .iter()
            .map(|i| { // prepare iterator for Device::new()
                (physical_device.queue_families().nth(*i as usize)
                    .unwrap_or_else(|| logger.error("FindQueueFamily", "Failed to find the requested queue family in the physical device")),
                queue_priority)
            }).collect::<Vec<_>>(); // we normally wouldnt need to collect but because iter is lazy it makes the borrow valid for the whole function

        let (device, mut queues) = Device::new(physical_device, &Features::none(),
            required_extensions, queue_families)
            .unwrap_or_else(|err| logger.error("CreateDevice", err));

        let graphics_queue = queues.next()
            .unwrap_or_else(|| logger.error("GetQueues", "No queues found"));
        let present_queue = queues.next()
            .unwrap_or_else(|| graphics_queue.clone()); // if theres no extra present queue just use the graphics queue

        (device, graphics_queue, present_queue)
    }

    fn dimensions(logger: &mut Logger, instance: &Arc<Instance>, surface: &Arc<Surface<Window>>, physical_device_index: usize) -> [u32; 2] {
        let physical_device = PhysicalDevice::from_index(instance, physical_device_index)
            .unwrap_or_else(|| logger.error("ReconstructPhysical", "Failed to reconstruct physical device from earlier obtained index"));
        let capabilities = surface.capabilities(physical_device)
            .unwrap_or_else(|err| match err {
                CapabilitiesError::OomError(err) => logger.error("SurfaceCapabilities", err),
                _ => logger.error("SurfaceCapabilities", err),
            });
        
        if let Some(current_extent) = capabilities.current_extent {
            // we got the current extent, use it
            current_extent
        } else {
            // get the current window size, we dont want a constant resolution
            let mut extent: (u32, u32) = (*surface).window().get_inner_size()
                .unwrap_or_else(|| logger.error("GetWindowSize", "Failed to get the current window dimensions"))
                .into();

            // make sure the extent is supported
            extent.0 = capabilities.min_image_extent[0]
                .max(capabilities.max_image_extent[0].min(extent.0));
            extent.1 = capabilities.min_image_extent[1]
                .max(capabilities.max_image_extent[1].min(extent.1));

            [extent.0, extent.1]
        }
    }

    fn swap_chain(
        logger: &mut Logger,
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        physical_device_index: usize,
        device: &Arc<Device>,
        graphics_queue: &Arc<Queue>,
        present_queue: &Arc<Queue>,
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        let physical_device = PhysicalDevice::from_index(instance, physical_device_index)
            .unwrap_or_else(|| logger.error("ReconstructPhysical", "Failed to reconstruct physical device from earlier obtained index"));
        let capabilities = surface.capabilities(physical_device)
            .unwrap_or_else(|err| match err {
                CapabilitiesError::OomError(err) => logger.error("SurfaceCapabilities", err),
                _ => logger.error("SurfaceCapabilities", err),
            });

        // TODO: temp test
        let surface_format = {
            /*capabilities.supported_formats.iter()
            // we try to find our preferred format
            .find(|(format, color_space)|
                *format == Format::B8G8R8A8Unorm && *color_space == ColorSpace::SrgbNonLinear
            )
            // if that fails, we just use the first supported format
            .unwrap_or(&capabilities.supported_formats[0])*/
            capabilities.supported_formats[0]
        };

        let present_mode = { // https://docs.rs/vulkano/0.10.0/vulkano/swapchain/enum.PresentMode.html
            let present_modes = capabilities.present_modes;

            if present_modes.mailbox {
                PresentMode::Mailbox
            } else {
                logger.warning("PresentMode", "Using VSync (Fifo), because Mailbox isn't supported");
                PresentMode::Fifo
            }
        };

        let dimensions = Self::dimensions(logger, instance, surface, physical_device_index);

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

        let sharing_mode: SharingMode = { // https://docs.rs/vulkano/0.10.0/vulkano/sync/enum.SharingMode.html
            let indices = required_queue_families(logger, surface, &physical_device);

            if indices.graphics != indices.present {
                vec![graphics_queue, present_queue].as_slice().into() // the queues are different, use both
            } else { graphics_queue.into() } // theres no present queue, just use the graphics queue
        };

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
            SwapchainCreationError::OomError(err) => logger.error("CreateSwapchain", err),
            _ => logger.error("CreateSwapchain", err),
        })
    }

    fn render_pass(
        logger: &mut Logger,
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
            RenderPassCreationError::OomError(err) => logger.error("CreateRenderPass", err),
            _ => logger.error("CreateRenderPass", err),
        }))
    }

    fn pipelines(
        logger: &mut Logger,
        device: &Arc<Device>,
        swap_chain_extent: [u32; 2],
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
    ) -> (pipelines::main::Pipeline) {
        //(
            pipelines::main::Pipeline::new(logger, &device, swap_chain_extent, &render_pass)//,
        //)
    }

    fn depth_buffer(
        logger: &mut Logger,
        device: &Arc<Device>,
        dimensions: [u32; 2],
    ) -> Arc<AttachmentImage<D16Unorm>> {
        AttachmentImage::transient(
            device.clone(),
            dimensions,
            D16Unorm,
        ).unwrap_or_else(|err| match err {
            ImageCreationError::AllocError(err) => match err {
                DeviceMemoryAllocError::OomError(err) => logger.error("CreateDepthBuffer", err),
                _ => logger.error("CreateDepthBuffer", err),
            },
            _ => logger.error("CreateDepthBuffer", err),
        })
    }

    fn framebuffers(
        logger: &mut Logger,
        swap_chain_images: &Vec<Arc<SwapchainImage<Window>>>,
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
        depth_buffer: &Arc<AttachmentImage<D16Unorm>>
    ) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
        swap_chain_images.iter()
            .map(|image| { // https://docs.rs/vulkano/0.10.0/vulkano/framebuffer/struct.FramebufferBuilder.html
                let framebuffer: Arc<FramebufferAbstract + Send + Sync> = Arc::new(Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                        .unwrap_or_else(|err| match err {
                            FramebufferCreationError::OomError(err) => logger.error("AddSwapchainImage", err),
                            _ => logger.error("AddSwapchainImage", err),
                        })
                    .add(depth_buffer.clone())
                        .unwrap_or_else(|err| match err {
                            FramebufferCreationError::OomError(err) => logger.error("AddDepthBuffer", err),
                            _ => logger.error("AddDepthBuffer", err),
                        })
                    .build()
                    .unwrap_or_else(|err| match err {
                        FramebufferCreationError::OomError(err) => logger.error("CreateFramebuffer", err),
                        FramebufferCreationError::AttachmentDimensionsIncompatible {expected, obtained} => logger.error("CreateFramebuffer", 
                            format!("{}; expected {:?}, got {:?}", err, expected, obtained)),
                        FramebufferCreationError::AttachmentsCountMismatch {expected, obtained} => logger.error("CreateFramebuffer",
                            format!("{}; expected {}, got {}", err, expected, obtained)),
                        _ => logger.error("CreateFramebuffer", err),
                    })
                ); framebuffer
            }
        ).collect::<Vec<_>>()
    }

    fn vertex_buffer(logger: &mut Logger, device: &Arc<Device>, vertices: Vec<Vertex>) -> Arc<BufferAccess + Send + Sync> {
        CpuAccessibleBuffer::from_iter(
            device.clone(), BufferUsage::vertex_buffer(), 
            vertices.iter().cloned())
            .unwrap_or_else(|err| match err {
                DeviceMemoryAllocError::OomError(err) => logger.error("CreateVertexBuffer", err),
                _ => logger.error("CreateVertexBuffer", err),
            })
    }

    fn normals_buffer(logger: &mut Logger, device: &Arc<Device>, normals: Vec<u16>) -> Arc<BufferAccess + Send + Sync> {
        CpuAccessibleBuffer::from_iter(
            device.clone(), BufferUsage::all(), 
            normals.iter().cloned())
            .unwrap_or_else(|err| match err {
                DeviceMemoryAllocError::OomError(err) => logger.error("CreateVertexBuffer", err),
                _ => logger.error("CreateVertexBuffer", err),
            })
    }

    fn index_buffer(logger: &mut Logger, device: &Arc<Device>, indices: Vec<u16>) -> Arc<BufferAccess + Send + Sync> {
        CpuAccessibleBuffer::from_iter(
            device.clone(), BufferUsage::index_buffer(), 
            indices.iter().cloned())
            .unwrap_or_else(|err| match err {
                DeviceMemoryAllocError::OomError(err) => logger.error("CreateIndexBuffer", err),
                _ => logger.error("CreateIndexBuffer", err),
            })
    }

    fn recreate_swap_chain(&mut self, logger: &mut Logger) -> bool {
        let dimensions = Self::dimensions(logger, &self.instance, &self.surface, self.physical_device);

        let (swap_chain, swap_chain_images) = match self.swap_chain.recreate_with_dimension(dimensions) {
            Ok(r) => r,
            Err(SwapchainCreationError::OomError(err)) => logger.error("CreateSwapchain", err),
            Err(SwapchainCreationError::UnsupportedDimensions) => return true,
            Err(err) => logger.error("CreateSwapchain", err),
        };
        self.swap_chain = swap_chain;
        self.swap_chain_images = swap_chain_images;

        self.render_pass = Self::render_pass(logger, &self.device, self.swap_chain.format());
        self.main_pipeline = Self::pipelines(logger, &self.device, self.swap_chain.dimensions(), &self.render_pass);

        self.depth_buffer = Self::depth_buffer(logger, &self.device, self.swap_chain.dimensions());
        self.swap_chain_framebuffers = Self::framebuffers(logger, &self.swap_chain_images, &self.render_pass, &self.depth_buffer);

        self.recreate_swap_chain = false;

        false
    }

    pub fn draw(&mut self, logger: &mut Logger, /*objects: Vec<Vec<Vertex>>*/) -> bool {
        if let Some(ref mut prev_frame) = self.prev_frame_end {
            prev_frame.cleanup_finished();
        }

        if self.recreate_swap_chain {
            if self.recreate_swap_chain(logger) { return true; }
        };

        let (image_index, acquire_future) = match swapchain::acquire_next_image(self.swap_chain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                self.recreate_swap_chain = true;
                return true;
            },
            Err(err) => logger.error("AcquireNextImage", err),
        };

        // drawing didn't fail
        false
    }   
}