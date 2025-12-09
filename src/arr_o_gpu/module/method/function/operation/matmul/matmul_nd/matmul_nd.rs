use std::sync::Arc;

use wgpu::{
    ComputePassDescriptor,
    ComputePipelineDescriptor,
    PipelineCompilationOptions,
    PipelineLayoutDescriptor,
    ShaderModuleDescriptor,
    ShaderSource,
    wgt::{ CommandEncoderDescriptor, PollType },
};

use crate::{
    ArrOgpuErr,
    ArrOgpuModule,
    GpuArray,
    arr_o_gpu::module::method::function::operation::matmul::matmul_nd_group_binding,
};

impl ArrOgpuModule {
    pub fn matmul_nd(&self, arr_a: &GpuArray, arr_b: &GpuArray) -> Result<GpuArray, ArrOgpuErr> {
        let shape_a = arr_a.shape();
        let shape_b = arr_b.shape();

        if shape_a.len() != shape_b.len() || shape_a.len() <= 1 || shape_b.len() <= 1 {
            let err = format!(
                "Array matmul nd Error, Array A {:?} can't matmul with Array B {:?}",
                shape_a,
                shape_b
            );

            return Err(ArrOgpuErr::MatmulND(err));
        }

        let k_a = shape_a[shape_a.len() - 1];
        let k_b = shape_b[shape_b.len() - 2];
        if
            shape_a.len() != shape_b.len() ||
            &shape_a[0..shape_a.len() - 2] != &shape_b[0..shape_a.len() - 2] ||
            k_a != k_b
        {
            let err = format!(
                "Array matmul nd Error, Array A {:?} can't matmul with Array B {:?}",
                shape_a,
                shape_b
            );

            return Err(ArrOgpuErr::MatmulND(err));
        }

        if shape_a.len() == 2 && shape_b.len() == 2 {
            return self.matmul_2d(arr_a, arr_b);
        }

        let wgpu_init = self.wgpu_init.read().unwrap();
        // bind group
        let (bind_group_layout, bind_group, output_shape, stride_out, allocate_out) =
            matmul_nd_group_binding(&wgpu_init.device, &self.allocator, arr_a, arr_b);
        let heap_bindgroup = &self.binding_compounds.read().unwrap()[0];

        // pipeline
        let pipeline_layout = wgpu_init.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Matmul ND"),
                bind_group_layouts: &[
                    // group 0
                    &heap_bindgroup.binding_group_layouts,
                    // group 1
                    &bind_group_layout,
                ],
                push_constant_ranges: &[],
            })
        );

        let shader = wgpu_init.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shaders For Matmul ND"),
            source: ShaderSource::Wgsl(include_str!("./matmul_nd.wgsl").into()),
        });

        let pipeline = wgpu_init.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline Layout For Matmul ND"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                layout: Some(&pipeline_layout),
                module: &shader,
            })
        );

        let mut encoder = wgpu_init.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("Create Encoder For Matmul ND"),
            })
        );

        {
            let mut bcp = encoder.begin_compute_pass(
                &(ComputePassDescriptor {
                    label: Some("Create Compute Pass For Matmul ND"),
                    timestamp_writes: None,
                })
            );

            bcp.set_pipeline(&pipeline);

            // heap
            bcp.set_bind_group(0, Some(&heap_bindgroup.binding_groups), &[]);
            // matmul nd
            bcp.set_bind_group(1, Some(&bind_group), &[]);

            let shape_of_matrix_a = &arr_a.shape[arr_a.shape.len() - 2..];
            let shape_of_matrix_b = &arr_b.shape[arr_b.shape.len() - 2..];

            let x = ((shape_of_matrix_a[0] as f32) / 16.0).ceil() as u32;
            let y = ((shape_of_matrix_b[1] as f32) / 16.0).ceil() as u32;
            let z = arr_a.shape[..arr_a.shape.len() - 2].iter().product::<u32>();
            bcp.dispatch_workgroups(x, y, z);
        }

        wgpu_init.queue.submit(Some(encoder.finish()));
        wgpu_init.device.poll(PollType::Wait).unwrap();

        let arr = GpuArray {
            module: Arc::new(self.clone()),
            pointer: (allocate_out.1, allocate_out.2),
            length: output_shape.iter().product::<u32>() as usize,
            stride: stride_out,
            shape: output_shape,
            space_type: allocate_out.0,
            binding: None,
        };

        Ok(arr)
    }
}
