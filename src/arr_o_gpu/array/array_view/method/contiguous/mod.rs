use std::num::NonZeroU64;

use wgpu::{ComputePipelineDescriptor, ShaderModuleDescriptor, ShaderSource};

use crate::{
    ArrOgpuErr, ArrayCompute, GpuArray, GpuArrayView, PipelineCompound,
    arr_o_gpu::compute_shaders::CONTIGUOUS_SHADERS_PATH, get_stride_from_shape, vector_padding,
};

const PIPELINE_CONTIGUOUS: &'static str = "pipeline_contiguous";

impl<'a, A> GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    pub fn contiguous(self) -> Result<GpuArray, ArrOgpuErr> {
        let module = self.module().clone();
        let mut allocator = self.array.module().allocator.write().unwrap();
        let wgpu = self.array.module().wgpu_module.read().unwrap();

        // output metadata
        let len = self.len();
        let shape = self.shape().clone();
        let stride = get_stride_from_shape(&shape);
        let offset = 0;
        let allocated = allocator
            .allocate(len)
            .map_err(|msg| ArrOgpuErr::Allocate(msg))?;
        let pointer = allocated.get_range();

        // output_metadata_buffer
        let pointer_arr = [pointer.0 as u32, pointer.1 as u32];
        let dim = shape.len();
        let padding_shape: [u32; 8] = vector_padding(shape.clone(), 0, 8)
            .map_err(|err| ArrOgpuErr::Contiguous(err))?
            .try_into()
            .unwrap();
        let padding_origin_stride: [u32; 8] = vector_padding(stride.clone(), 0, 8)
            .map_err(|err| ArrOgpuErr::Contiguous(err))?
            .try_into()
            .unwrap();
        let output_metadata = module.create_metadata_compound(
            pointer_arr,
            len,
            dim as u32,
            offset,
            padding_shape,
            padding_origin_stride,
            padding_origin_stride,
        );

        // bind group
        let heap_bind = module.heap_binding();

        // pipeline
        let read = module.pipeline_cache.read().unwrap();
        let pipeline = if let Some(pipeline) = read.get(PIPELINE_CONTIGUOUS) {
            PipelineCompound::PipelineCache(pipeline)
        } else {
            let pipeline = wgpu
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("Create Pipeline Layout For Contiguous"),
                    layout: Some(&module.common_pipeline_layout),
                    module: &wgpu.device.create_shader_module(ShaderModuleDescriptor {
                        label: Some("Create Shader Module For Contiguous"),
                        source: ShaderSource::Wgsl(CONTIGUOUS_SHADERS_PATH.into()),
                    }),
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[],
                        zero_initialize_workgroup_memory: false,
                    },
                    cache: None,
                });

            PipelineCompound::UnsavePipeline(pipeline, PIPELINE_CONTIGUOUS)
        };

        // encoder
        let mut encoder =
            wgpu.device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Create Encoder For Contiguous"),
                });

        // // set metadata
        // // // execute_cache
        let execute_cache = &module.execute_args;
        // // // metadata a
        let array_metadata = self.metadata_compound.as_ref().unwrap();
        encoder.copy_buffer_to_buffer(&array_metadata.buffer, 0, &execute_cache, 0, 256);
        encoder.copy_buffer_to_buffer(&output_metadata.buffer, 0, &execute_cache, 256, 256);

        {
            let mut begin_compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Create Begin Compute Pass"),
                timestamp_writes: None,
            });

            pipeline.set_pipeline_begin_compute_pass(&mut begin_compute_pass);

            begin_compute_pass.set_bind_group(0, Some(&heap_bind.binding_groups), &[]);

            let x = (len + 255) / 256;
            begin_compute_pass.dispatch_workgroups(x, 1, 1);
        }

        wgpu.queue.submit(Some(encoder.finish()));

        let unsave = pipeline.get_unsave_pipeline();
        drop(read);
        module.saving_from_pipeline_compound(unsave);

        let array = GpuArray {
            module,
            length: len as usize,
            metadata_compound: Some(output_metadata),
            shape,
            stride,
            allocated,
        };

        Ok(array)
    }
}
