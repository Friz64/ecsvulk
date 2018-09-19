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

                    let pipeline = Arc::new(GraphicsPipeline::start()
                        .vertex_input_single_buffer::<Vertex>()
                        .vertex_shader(vert.main_entry_point(), ())
                        .triangle_list()
                        .viewports_dynamic_scissors_irrelevant(1)
                        .fragment_shader(frag.main_entry_point(), ())
                        .depth_stencil_simple_depth()
                        .render_pass(Subpass::from(render_pass.clone(), 0)
                            .unwrap_or_else(|| logger.error("CreatePipeline", "Failed to create Subpass")))
                        .build(device.clone())
                            .unwrap_or_else(|err| logger.error("CreatePipeline", err))
                    );
                    
                    let cbp = CpuBufferPool::new(device.clone(), BufferUsage::all());

                    $name::Pipeline {
                        pipeline,
                        cbp,
                    }
                }
            }
        }
    };
}