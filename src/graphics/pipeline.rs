#[derive(Copy, Clone)]
pub struct PipelineSettings {
    pub backfaceculling: bool,
    pub wireframe: bool,
}

macro_rules! gen_pipeline {
    ($name:ident, $vert_src:expr, $frag_src:expr) => {
        pub mod $name {
            pub mod vs {
                #[derive(VulkanoShader)]
                #[ty = "vertex"]
                #[path = $vert_src]
                struct _Shdr; 
            }
            pub mod fs {
                #[derive(VulkanoShader)]
                #[ty = "fragment"]
                #[path = $frag_src]
                struct _Shdr;
            }

            use super::*;

            pub struct Pipeline {
                pub pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
                pub cbp: CpuBufferPool<vs::ty::Data>,
                pub sets_pool: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract + Send + Sync>>,
                pub settings: PipelineSettings,
            }

            impl Pipeline {
                pub fn new(
                    logger: &mut Logger,
                    device: &Arc<Device>,
                    render_pass: &Arc<RenderPassAbstract + Send + Sync>,
                    settings: PipelineSettings,
                ) -> Self {
                    let vert = $name::vs::Shader::load(device.clone())
                        .unwrap_or_else(|err| logger.error("ShaderLoad", err));
                    let frag = $name::fs::Shader::load(device.clone())
                        .unwrap_or_else(|err| logger.error("ShaderLoad", err));

                    let tbd: TwoBuffersDefinition<Vertex, Normal> = TwoBuffersDefinition::new();

                    let mut pipeline = GraphicsPipeline::start()
                        .vertex_input(tbd)
                        .vertex_shader(vert.main_entry_point(), ())
                        .viewports_dynamic_scissors_irrelevant(1)
                        .fragment_shader(frag.main_entry_point(), ())
                        .triangle_list()
                        .depth_stencil_simple_depth();

                    if settings.backfaceculling {
                        pipeline = pipeline.cull_mode_back();
                    }
                    if settings.wireframe {
                        pipeline = pipeline.polygon_mode_line();
                    }
                    
                    let pipeline = Arc::new(pipeline
                        .render_pass(Subpass::from(render_pass.clone(), 0)
                            .unwrap_or_else(|| logger.error("CreatePipeline", "Failed to create Subpass")))
                        .build(device.clone())
                            .unwrap_or_else(|err| logger.error(&("CreatePipeline_".to_owned() + stringify!($name)), err))
                    );
                    
                    let cbp = CpuBufferPool::new(device.clone(), BufferUsage::all());
                    let sets_pool: FixedSizeDescriptorSetsPool<
                        Arc<GraphicsPipelineAbstract + Send + Sync>
                    > = FixedSizeDescriptorSetsPool::new(pipeline.clone(), 0);

                    $name::Pipeline {
                        pipeline: pipeline,
                        cbp,
                        sets_pool,
                        settings,
                    }
                }
            }
        }
    };
}