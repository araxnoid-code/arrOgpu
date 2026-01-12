use wgpu::{ComputePass, ComputePipeline};

use crate::ArrOgpuModule;

pub enum PipelineCompound<'a> {
    PipelineCache(&'a ComputePipeline),
    UnsavePipeline(ComputePipeline, &'static str),
}

impl<'a> PipelineCompound<'a> {
    pub fn set_pipeline_begin_compute_pass(&self, begin_compute_pass: &mut ComputePass) {
        match self {
            Self::PipelineCache(cache) => begin_compute_pass.set_pipeline(cache),
            Self::UnsavePipeline(pipeline, _) => begin_compute_pass.set_pipeline(pipeline),
        }
    }

    pub fn get_unsave_pipeline(self) -> Option<(ComputePipeline, &'static str)> {
        if let Self::UnsavePipeline(pipeline, key) = self {
            Some((pipeline, key))
        } else {
            None
        }
    }
}

impl ArrOgpuModule {
    pub(crate) fn saving_from_pipeline_compound(
        &self,
        unsave_pipeline: Option<(ComputePipeline, &'static str)>,
    ) {
        if let Some((pipeline, key)) = unsave_pipeline {
            let mut write = self.pipeline_cache.write().unwrap();
            write.insert(key, pipeline);
        }
    }
}
