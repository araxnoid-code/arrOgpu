use std::sync::Arc;

use wgpu::{
    ComputePassDescriptor,
    ComputePipelineDescriptor,
    PipelineLayoutDescriptor,
    ShaderModuleDescriptor,
    ShaderSource,
    wgt::{ CommandEncoderDescriptor, PollType },
};

use crate::{ ArrayView, GpuArray, GpuArrayView, bind_group_collect, get_stride_from_shape };

impl<'a, A> GpuArrayView<'a, A> where A: ArrayView {
    pub fn contiguous(self) -> GpuArray {
        let mut allocator = self.array.module().allocator.write().unwrap();
        let wgpu = self.array.module().wgpu_init.read().unwrap();

        // array
        // // pointer
        let pointer = self.array.pointer_to_arr();

        // // shape
        let shape = &self.shape;

        // // iters
        let iters = get_stride_from_shape(&shape);

        // // stride
        let stride = &self.stride;

        // // offset
        let offset = self.offset;

        // // product
        let len = shape.iter().product::<u32>();

        // output
        let allocate = allocator.pointer_input(len);
        // // pointer
        let pointer_out = [allocate.1, allocate.2];

        // bind group
        let heap_binding = &self.array.module().binding_compounds.read().unwrap()[0];
        let (bind_group_layout, bind_group) = bind_group_collect(
            &wgpu,
            &pointer,
            shape,
            &iters,
            stride,
            &offset,
            &len,
            &pointer_out
        );

        // pipeline
        let pipeline_layout = wgpu.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Collect View"),
                bind_group_layouts: &[
                    // heap
                    &heap_binding.binding_group_layouts,
                    // array
                    &bind_group_layout,
                ],
                push_constant_ranges: &[],
            })
        );

        let shader = wgpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shaders For Collect View"),
            source: ShaderSource::Wgsl(include_str!("./view_collect.wgsl").into()),
        });

        let pipeline = wgpu.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline Layout For Collect View"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                layout: Some(&pipeline_layout),
                module: &shader,
            })
        );

        // encoder
        let mut encoder = wgpu.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("Create Encoder For Collect View"),
            })
        );

        {
            let mut bcp = encoder.begin_compute_pass(
                &(ComputePassDescriptor {
                    label: Some("Create Begin Compute Pass For Collect View"),
                    timestamp_writes: None,
                })
            );

            bcp.set_pipeline(&pipeline);
            // heap
            bcp.set_bind_group(0, Some(&heap_binding.binding_groups), &[]);
            // array
            bcp.set_bind_group(1, Some(&bind_group), &[]);

            let x = (len + 256 - 1) / 256;
            bcp.dispatch_workgroups(x, 1, 1);
        }

        wgpu.queue.submit(Some(encoder.finish()));
        wgpu.device.poll(PollType::Wait).unwrap();

        let array = GpuArray {
            module: self.array.module().clone(),
            length: len as usize,
            pointer: (allocate.1, allocate.2),
            space_type: allocate.0,
            stride: get_stride_from_shape(&shape),
            shape: self.shape,
            binding: None,
        };

        array
    }
}
