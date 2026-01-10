use wgpu::{ComputePass, ComputePipeline};

pub enum PipelineCompound<'a> {
    PipelineCache(&'a ComputePipeline),
    Pipeline(ComputePipeline, &'static str),
}

impl<'a> PipelineCompound<'a> {
    pub fn set_pipeline_begin_compute_pass(&self, begin_compute_pass: &mut ComputePass) {
        match self {
            Self::PipelineCache(cache) => begin_compute_pass.set_pipeline(cache),
            Self::Pipeline(pipeline, _) => begin_compute_pass.set_pipeline(pipeline),
        }
    }
}
