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
            }

            impl Pipeline {
                pub fn new(
                    logger: &mut Logger,
                    device: &Arc<Device>,
                    render_pass: &Arc<RenderPassAbstract + Send + Sync>,
                ) -> Self {
                    let vert = $name::vs::Shader::load(device.clone())
                        .unwrap_or_else(|err| logger.error("ShaderLoad", err));
                    let frag = $name::fs::Shader::load(device.clone())
                        .unwrap_or_else(|err| logger.error("ShaderLoad", err));

                    let pipeline: Arc<
                        GraphicsPipeline<
                            TwoBuffersDefinition<Vertex, Normal>,
                            Box<dyn PipelineLayoutAbstract + marker::Sync + marker::Send>,
                            Arc<dyn RenderPassAbstract + marker::Sync + marker::Send>
                        >
                    > = Arc::new(GraphicsPipeline::start()
                        .vertex_input(TwoBuffersDefinition::new())
                        .vertex_shader(vert.main_entry_point(), ())
                        .triangle_list()
                        .cull_mode_back()
                        // wireframe
                        //.polygon_mode_line()

                        .viewports_dynamic_scissors_irrelevant(1)
                        .fragment_shader(frag.main_entry_point(), ())
                        .depth_stencil_simple_depth()
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
                    }
                }
            }
        }
    };
}