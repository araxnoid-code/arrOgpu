use std::sync::Arc;

use wgpu::{
    CommandEncoderDescriptor,
    ComputePassDescriptor,
    ComputePipelineDescriptor,
    PipelineCompilationOptions,
    PipelineLayoutDescriptor,
    ShaderModuleDescriptor,
    ShaderSource,
};

use crate::{ ArrOgpuErr, ArrOgpuModule, ArrayView, GpuArray, get_stride_from_shape };

impl ArrOgpuModule {
    pub fn mul_view<'a, A, B>(&self, array_a: &A, array_b: &B) -> Result<GpuArray, ArrOgpuErr>
        where A: ArrayView, B: ArrayView
    {
        if array_a.shape() != array_b.shape() {
            let err = format!(
                "Array Div Error, Shape Of A is {:?} But Multiplied with Shape Of B is {:?}",
                array_a.shape(),
                array_b.shape()
            );
            return Err(ArrOgpuErr::Add(err));
        }

        let len = array_a.len();
        let shape = array_a.shape();
        let stride = get_stride_from_shape(&shape);
        let allocate = self.allocator.write().unwrap().pointer_input(len);
        let out_binding = self.array_data_binding(
            &[allocate.1, allocate.2],
            shape,
            &stride,
            &stride,
            &0
        );

        let wgpu = self.wgpu_init.read().unwrap();

        let heap_binding = &self.heap_binding;
        let array_a_binding = array_a.binding();
        let array_b_binding = array_b.binding();
        let pipeline_layout = wgpu.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Mul"),
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
            })
        );

        let shader = wgpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shader For Mul"),
            source: ShaderSource::Wgsl(include_str!("mul.wgsl").into()),
        });
        let pipeline = wgpu.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline For Mul"),
                cache: None,
                layout: Some(&pipeline_layout),
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                module: &shader,
            })
        );

        let mut encoder = wgpu.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("Create Encoder For Mul"),
            })
        );

        {
            let mut bcp = encoder.begin_compute_pass(
                &(ComputePassDescriptor {
                    label: Some("Create Compute Pass For Mul"),
                    timestamp_writes: None,
                })
            );

            bcp.set_pipeline(&pipeline);
            // heap
            bcp.set_bind_group(0, Some(&heap_binding.binding_groups), &[]);

            // Array A
            bcp.set_bind_group(1, Some(&array_a_binding.1), &[]);

            // Array B
            bcp.set_bind_group(2, Some(&array_b_binding.1), &[]);

            // Out
            bcp.set_bind_group(3, Some(&out_binding.1), &[]);

            let x = ((len as f32) / 256.0).ceil() as u32;
            bcp.dispatch_workgroups(x, 1, 1);
        }

        wgpu.queue.submit(Some(encoder.finish()));
        wgpu.device.poll(wgpu::wgt::PollType::Wait).unwrap();

        let array = GpuArray {
            module: Arc::new(self.clone()),
            binding: out_binding,
            length: len as usize,
            pointer: (allocate.1, allocate.2),
            shape: shape.clone(),
            stride,
            space_type: allocate.0,
        };

        Ok(array)
    }
}
