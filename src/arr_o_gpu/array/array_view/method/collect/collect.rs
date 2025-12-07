use wgpu::{
    ComputePassDescriptor,
    ComputePipelineDescriptor,
    PipelineLayoutDescriptor,
    ShaderModuleDescriptor,
    ShaderSource,
    wgt::{ CommandEncoderDescriptor, PollType },
};

use crate::{ GpuArrayView, bind_group_collect };

impl<'a> GpuArrayView<'a> {
    pub fn collect(&self) {
        let mut allocator = self.array.module.allocator.write().unwrap();
        let wgpu = self.array.module.wgpu_init.read().unwrap();

        // array
        // // pointer
        let pointer = self.array.pointer_to_arr();

        // // shape
        let shape = &self.shape;

        // // stride
        let stride = self.array.stride();

        // // offset
        let offset = self.offset;

        // // product
        let len = shape.iter().product::<u32>();

        // output
        let allocate = allocator.pointer_input(len);
        // // pointer
        let pointer_out = [allocate.1, allocate.2];

        // bind group
        let heap_binding = &self.array.module.binding_compounds.read().unwrap()[0];
        let (bind_group_layout, bind_group) = bind_group_collect(
            &wgpu,
            &pointer,
            shape,
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
    }
}
