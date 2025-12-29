use std::sync::Arc;

use wgpu::{
    ComputePassDescriptor, ComputePipelineDescriptor, PipelineLayoutDescriptor,
    ShaderModuleDescriptor,
};

use crate::{ArrOgpuErr, ArrOgpuModule, ArrayCompute, GpuArray, get_stride_from_shape};

impl ArrOgpuModule {
    pub fn sin<A>(&self, array: &A) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
    {
        let wgpu = self.wgpu_init.read().unwrap();

        // out meta data
        let shape = array.shape();
        let stride = get_stride_from_shape(&shape);
        let len = array.len();
        let allocate = self.allocator.write().unwrap().pointer_input(len);

        // binding
        // // heap
        let heap_bind = &self.heap_binding;
        // // array
        let array_bind = array.binding();
        // // out
        let out_bind =
            self.create_metadata_binding(&[allocate.1, allocate.2], shape, &stride, &stride, &0);

        // pipeline
        let pipeline_layout = wgpu.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Sin"),
                bind_group_layouts: &[&heap_bind.binding_group_layouts, &array_bind.0, &out_bind.0],
                immediate_size: 0,
            }),
        );

        let shader = wgpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shader Module For Sin"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./wgsl/sin.wgsl").into()),
        });

        let pipeline = wgpu.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline For Sin"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                layout: Some(&pipeline_layout),
                module: &shader,
            }),
        );

        // encoder
        let mut encoder = wgpu.device.create_command_encoder(
            &(wgpu::wgt::CommandEncoderDescriptor {
                label: Some("Create Encoder For Sin"),
            }),
        );

        {
            // begin compute pass
            let mut bcp = encoder.begin_compute_pass(
                &(ComputePassDescriptor {
                    label: Some("Create Begin Compute Pass For Sin"),
                    timestamp_writes: None,
                }),
            );

            // set pipeline
            bcp.set_pipeline(&pipeline);

            // bind_group
            bcp.set_bind_group(0, Some(&heap_bind.binding_groups), &[]);
            bcp.set_bind_group(1, Some(&array_bind.1), &[]);
            bcp.set_bind_group(2, Some(&out_bind.1), &[]);

            let x = (len + 256 - 1) / 256;
            bcp.dispatch_workgroups(x, 1, 1);
        }
        wgpu.queue.submit(Some(encoder.finish()));
        // if let Err(err_poll) = wgpu.device.poll(wgpu::wgt::PollType::Wait) {
        //     let error = "Add Error, Error While Poll".to_string();
        //     return Err(ArrOgpuErr::Poll(error, err_poll));
        // }

        let array = GpuArray {
            // build/0.1.0.5
            metadata_compound: None,
            // build/0.1.0.5
            module: Arc::new(self.clone()),
            binding: out_bind,
            length: len as usize,
            pointer: (allocate.1, allocate.2),
            shape: shape.clone(),
            space_type: allocate.0,
            stride,
        };

        Ok(array)
    }
}
