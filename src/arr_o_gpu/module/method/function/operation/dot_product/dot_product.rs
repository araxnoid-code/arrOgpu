use std::sync::Arc;

use wgpu::{
    ComputePassDescriptor, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
};

use crate::{ArrOgpuErr, ArrOgpuModule, ArrayCompute, GpuArray};

impl ArrOgpuModule {
    pub fn dot_product<A, B>(&self, array_a: &A, array_b: &B) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
        B: ArrayCompute,
    {
        let shape_a = array_a.shape();
        let shape_b = array_b.shape();

        let length_a = array_a.len();
        let length_b = array_b.len();

        if shape_a.len() != 1 || shape_b.len() != 1 || length_a != length_b {
            let err = format!(
                "Dot Product Error, Array with shape {:?} and With Shape {:?} Can't Be Operated",
                shape_a, shape_b
            );
            return Err(ArrOgpuErr::DotProduct(err));
        }

        let shape = vec![1];
        let len = 1;
        let stride = vec![1];
        let allocate = self.allocator.write().unwrap().pointer_input(len);

        let wgpu = self.wgpu_init.read().unwrap();

        // binding
        let heap_binding = &self.heap_binding;
        let array_a_binding = array_a.binding();
        let array_b_binding = array_b.binding();
        let out_binding =
            self.create_metadata_binding(&[allocate.1, allocate.2], &shape, &stride, &stride, &0);

        // // pipeline_layout
        let pipeline_layout = wgpu.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Dot Product"),
                bind_group_layouts: &[
                    // heap
                    &heap_binding.binding_group_layouts,
                    // array a
                    &array_a_binding.0,
                    // array b
                    &array_b_binding.0,
                    // output
                    &out_binding.0,
                ],
                push_constant_ranges: &[],
            }),
        );

        // // pipeline
        let shader = wgpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shader For Dot Product"),
            source: ShaderSource::Wgsl(include_str!("dot_product_type_a.wgsl").into()),
        });
        let pipeline = wgpu.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline For Matmul 2D"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                layout: Some(&pipeline_layout),
                module: &shader,
            }),
        );

        // let encoder
        let mut encoder = wgpu.device.create_command_encoder(
            &(wgpu::wgt::CommandEncoderDescriptor {
                label: Some("Create Encoder For Dot Product"),
            }),
        );

        {
            // begin compute pass
            let mut bcp = encoder.begin_compute_pass(
                &(ComputePassDescriptor {
                    label: Some("Create Begin COmpute Pass For Dot Product"),
                    timestamp_writes: None,
                }),
            );

            // pipeline
            bcp.set_pipeline(&pipeline);

            // group
            bcp.set_bind_group(0, Some(&heap_binding.binding_groups), &[]);
            bcp.set_bind_group(1, Some(&array_a_binding.1), &[]);
            bcp.set_bind_group(2, Some(&array_b_binding.1), &[]);
            bcp.set_bind_group(3, Some(&out_binding.1), &[]);

            // dispact
            bcp.dispatch_workgroups(1, 1, 1);
        }
        wgpu.queue.submit(Some(encoder.finish()));
        wgpu.device.poll(wgpu::wgt::PollType::Wait).unwrap();

        let array = GpuArray {
            module: Arc::new(self.clone()),
            shape,
            pointer: (allocate.1, allocate.2),
            length: len as usize,
            space_type: allocate.0,
            stride,
            binding: out_binding,
        };

        Ok(array)
    }
}
