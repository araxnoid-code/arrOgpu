use std::sync::Arc;

use wgpu::{
    ComputePassDescriptor,
    ComputePipelineDescriptor,
    PipelineCompilationOptions,
    PipelineLayoutDescriptor,
    ShaderModuleDescriptor,
    wgt::CommandEncoderDescriptor,
};

use crate::{ ArrOgpuErr, ArrOgpuModule, GpuArray, bind_group_dot_product };

impl ArrOgpuModule {
    pub fn dot_product(
        &self,
        array_a: &GpuArray,
        array_b: &GpuArray
    ) -> Result<GpuArray, ArrOgpuErr> {
        let shape_a = array_a.shape();
        let shape_b = array_b.shape();

        let length_a = array_a.len();
        let length_b = array_b.len();

        if shape_a.len() != 1 || shape_b.len() != 1 || length_a != length_b {
            let err = format!(
                "Array Dot Product Error, Array with shape {:?} and With Shape {:?} Can't Be Operated",
                shape_a,
                shape_b
            );
            return Err(ArrOgpuErr::DotProduct(err));
        }

        let length = length_a as u32;
        let pointer_a = array_a.pointer_to_arr();
        let pointer_b = array_b.pointer_to_arr();
        let output_allocate = self.allocator_write().pointer_input(1);

        let pointer_output = [output_allocate.1, output_allocate.2];

        let wgpu_init = self.wgpu_init().read().unwrap();

        // bind_group
        let heap_bind_group = &self.binding_compounds.read().unwrap()[0];
        let (bing_group, bind_group_layout) = bind_group_dot_product(
            &wgpu_init,
            length,
            &pointer_a,
            &pointer_b,
            &pointer_output
        );

        // pipeline and shader
        let pipeline_layout = wgpu_init.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Dot Product"),
                push_constant_ranges: &[],
                bind_group_layouts: &[
                    // heap
                    &heap_bind_group.binding_group_layouts,
                    &bind_group_layout,
                ],
            })
        );

        let shader = wgpu_init.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shaders For Dot Product"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./dot_product.wgsl").into()),
        });

        let pipeline = wgpu_init.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline Layout For Dot Product"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                layout: Some(&pipeline_layout),
                module: &shader,
            })
        );

        let mut encoder = wgpu_init.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("Create Encoder For Dot Product"),
            })
        );

        {
            let mut bcp = encoder.begin_compute_pass(
                &(ComputePassDescriptor {
                    label: Some("Create Compute Pass For Dot Product"),
                    timestamp_writes: None,
                })
            );

            bcp.set_pipeline(&pipeline);

            // group 0
            bcp.set_bind_group(0, Some(&heap_bind_group.binding_groups), &[]);
            // group 1
            bcp.set_bind_group(1, Some(&bing_group), &[]);
            bcp.dispatch_workgroups(1, 1, 1);
        }

        wgpu_init.queue.submit(Some(encoder.finish()));
        wgpu_init.device.poll(wgpu::wgt::PollType::Wait).unwrap();

        let shape = vec![1];
        let stride = vec![1];
        let binding = self.array_data_binding(
            &[output_allocate.1, output_allocate.2],
            &shape,
            &stride,
            &stride,
            &0
        );

        let arr = GpuArray {
            length: length_a,
            module: Arc::new(self.clone()),
            pointer: (output_allocate.1, output_allocate.2),
            shape,
            space_type: output_allocate.0,
            stride,
            binding: binding,
        };

        Ok(arr)
    }
}
