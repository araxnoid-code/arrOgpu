use std::sync::Arc;

use wgpu::{
    wgt::{ CommandEncoderDescriptor, PollType },
    BindGroup,
    BindGroupLayout,
    ComputePassDescriptor,
    ComputePipelineDescriptor,
    PipelineCompilationOptions,
    PipelineLayoutDescriptor,
    ShaderModuleDescriptor,
    ShaderSource,
};

use crate::{ get_stride_from_shape, ArrOgpuModule, GpuArray, SpaceType };

impl ArrOgpuModule {
    pub(crate) fn array_from_of_output(
        &self,
        shape: &[u32],
        pointer: (usize, usize),
        space_type: SpaceType,
        binding_of_output_layout: &BindGroupLayout,
        binding_of_output: &BindGroup
    ) -> GpuArray {
        // wgpu
        let wgpu_init = self.wgpu_init.read().unwrap();

        let shader = wgpu_init.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shader For Init Array Of Output"),
            source: ShaderSource::Wgsl(
                include_str!("./../../../shader/shaders/array_init_of_output.wgsl").into()
            ),
        });

        let pipeline_layout = wgpu_init.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Init Array Of Output"),
                bind_group_layouts: &[
                    &self.binding_compounds.read().unwrap()[0].binding_group_layouts,
                    binding_of_output_layout,
                ],
                push_constant_ranges: &[],
            })
        );

        let pipeline = wgpu_init.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline For Init Array Of Output"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                layout: Some(&pipeline_layout),
                module: &shader,
            })
        );

        let mut encoder = wgpu_init.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("Create Encoder For Init Array Of Output"),
            })
        );

        {
            let mut bcp = encoder.begin_compute_pass(
                &(ComputePassDescriptor {
                    label: Some("Create Compute Pass For Array Of Output"),
                    timestamp_writes: None,
                })
            );

            bcp.set_pipeline(&pipeline);

            // heap
            let heap = &self.binding_compounds.read().unwrap()[0].binding_groups;
            bcp.set_bind_group(0, Some(heap), &[]);
            bcp.set_bind_group(1, Some(binding_of_output), &[]);
        }

        wgpu_init.queue.submit(Some(encoder.finish()));
        wgpu_init.device.poll(PollType::Wait);

        let len = shape.iter().product::<u32>() as usize;
        let stride = get_stride_from_shape(shape);
        GpuArray {
            module: Arc::new(self.clone()),
            shape: shape.to_owned(),
            length: len,
            pointer,
            space_type,
            stride,
        }
    }
}
