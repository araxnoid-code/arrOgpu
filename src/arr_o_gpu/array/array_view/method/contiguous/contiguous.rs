use wgpu::{
    ComputePassDescriptor, ComputePipelineDescriptor, PipelineLayoutDescriptor,
    ShaderModuleDescriptor, ShaderSource,
    wgt::{CommandEncoderDescriptor, PollType},
};

use crate::{ArrayCompute, GpuArray, GpuArrayView, get_stride_from_shape};

impl<'a, A> GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    pub fn contiguous(self) -> GpuArray {
        let mut allocator = self.array.module().allocator.write().unwrap();
        let wgpu = self.array.module().wgpu_init.read().unwrap();

        // // shape
        let shape = &self.shape;

        // // iters
        let iters = get_stride_from_shape(&shape);

        // // product
        let len = shape.iter().product::<u32>();

        // output
        let allocate = allocator.pointer_input(len);
        // // pointer
        let pointer_out = [allocate.1, allocate.2];

        // bind group
        let heap_binding = &self.array.module().heap_binding;
        let array_bind_group = self.binding();
        let output_bind_group =
            self.module()
                .create_metadata_binding(&pointer_out, &self.shape, &iters, &iters, &0);

        // pipeline
        let pipeline_layout = wgpu.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Collect View"),
                bind_group_layouts: &[
                    // heap
                    &heap_binding.binding_group_layouts,
                    // array
                    // &bind_group_layout,
                    &array_bind_group.0,
                    // output
                    &output_bind_group.0,
                ],
                push_constant_ranges: &[],
            }),
        );

        let shader = wgpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shaders For Collect View"),
            source: ShaderSource::Wgsl(include_str!("./view_contiguous.wgsl").into()),
        });

        let pipeline = wgpu.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline Layout For Collect View"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                layout: Some(&pipeline_layout),
                module: &shader,
            }),
        );

        // encoder
        let mut encoder = wgpu.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("Create Encoder For Collect View"),
            }),
        );

        {
            let mut bcp = encoder.begin_compute_pass(
                &(ComputePassDescriptor {
                    label: Some("Create Begin Compute Pass For Collect View"),
                    timestamp_writes: None,
                }),
            );

            bcp.set_pipeline(&pipeline);
            // heap
            bcp.set_bind_group(0, Some(&heap_binding.binding_groups), &[]);
            // array
            bcp.set_bind_group(1, Some(&array_bind_group.1), &[]);
            // output
            bcp.set_bind_group(2, Some(&output_bind_group.1), &[]);

            let x = (len + 256 - 1) / 256;
            bcp.dispatch_workgroups(x, 1, 1);
        }

        wgpu.queue.submit(Some(encoder.finish()));
        wgpu.device.poll(PollType::Wait).unwrap();

        let stride = get_stride_from_shape(&shape);
        let binding = self.module().create_metadata_binding(
            &[allocate.1, allocate.2],
            &self.shape,
            &stride,
            &stride,
            &0,
        );

        let array = GpuArray {
            module: self.array.module().clone(),
            length: len as usize,
            pointer: (allocate.1, allocate.2),
            space_type: allocate.0,
            stride,
            shape: self.shape,
            binding,
        };

        array
    }
}
