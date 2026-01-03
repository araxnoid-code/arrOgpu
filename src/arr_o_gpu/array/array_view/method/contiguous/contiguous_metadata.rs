use std::sync::Arc;

use wgpu::{
    ComputePipelineDescriptor, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
};

use crate::{
    ArrOgpuErr, ArrayCompute, GpuArray, GpuArrayView, get_stride_from_shape, vector_padding,
};

impl<'a, A> GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    pub fn contiguous_metadata(self) -> Result<GpuArray, ArrOgpuErr> {
        let module = self.module().clone();
        let mut allocator = self.array.module().allocator.write().unwrap();
        let wgpu = self.array.module().wgpu_init.read().unwrap();

        // output metadata
        let len = self.len();
        let shape = self.shape().clone();
        let stride = get_stride_from_shape(&shape);
        let offset = 0;
        let allocate = allocator.pointer_input(len);

        // output_metadata_buffer
        let pointer_arr = [allocate.1, allocate.2];
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
        let shaders = wgpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shader Module For Contiguous"),
            source: ShaderSource::Wgsl(include_str!("contiguous_metadata.wgsl").into()),
        });

        let pipeline_layout = wgpu
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Contiguous"),
                bind_group_layouts: &[&heap_bind.binding_group_layouts],
                immediate_size: 0,
            });

        let pipeline = wgpu
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Create Pipeline Layout For Contiguous"),
                layout: Some(&pipeline_layout),
                module: &shaders,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("POINTER_START", self.pointer().0 as f64),
                        ("POINTER_START_OUTPUT", allocate.1 as f64),
                        ("LEN", len as f64),
                        ("DIM", self.dim() as f64),
                        ("OFFSET", self.offset() as f64),
                    ],
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });

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
        let array_metadata = self.metadata_compound.unwrap();
        encoder.copy_buffer_to_buffer(&array_metadata.buffer, 0, &execute_cache, 0, 256);

        {
            let mut begin_compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Create Begin Compute Pass"),
                timestamp_writes: None,
            });

            begin_compute_pass.set_pipeline(&pipeline);
            begin_compute_pass.set_bind_group(0, Some(&heap_bind.binding_groups), &[]);

            let x = (len + 255) / 256;
            begin_compute_pass.dispatch_workgroups(x, 1, 1);
        }

        wgpu.queue.submit(Some(encoder.finish()));

        let array = GpuArray {
            module,
            binding: None,
            length: len as usize,
            metadata_compound: Some(output_metadata),
            pointer: (allocate.1, allocate.2),
            space_type: allocate.0,
            shape,
            stride,
        };

        Ok(array)
    }
}
