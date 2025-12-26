use std::sync::Arc;

use wgpu::{
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
};

use crate::{ArrOgpuErr, ArrOgpuModule, ArrayCompute, GpuArray, get_stride_from_shape};

impl ArrOgpuModule {
    pub fn matmul_nd<A, B>(&self, array_a: &A, array_b: &B) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
        B: ArrayCompute,
    {
        let shape_a = array_a.shape();
        let shape_b = array_b.shape();

        if shape_a.len() != shape_b.len() || shape_a.len() <= 1 || shape_b.len() <= 1 {
            let err = format!(
                "Array matmul nd Error, Array A {:?} can't matmul with Array B {:?}",
                shape_a, shape_b
            );

            return Err(ArrOgpuErr::MatmulND(err));
        }

        let k_a = shape_a[shape_a.len() - 1];
        let k_b = shape_b[shape_b.len() - 2];
        if shape_a.len() != shape_b.len()
            || &shape_a[0..shape_a.len() - 2] != &shape_b[0..shape_a.len() - 2]
            || k_a != k_b
        {
            let err = format!(
                "Array matmul nd Error, Array A {:?} can't matmul with Array B {:?}",
                shape_a, shape_b
            );

            return Err(ArrOgpuErr::MatmulND(err));
        }

        if shape_a.len() == 2 && shape_b.len() == 2 {
            return self.matmul_2d(array_a, array_b);
        }

        let mut output_shape = shape_a.clone();
        *output_shape.last_mut().unwrap() = *shape_b.last().unwrap();
        let len = output_shape.iter().product::<u32>();
        let stride = get_stride_from_shape(&output_shape);
        let allocate = self.allocator.write().unwrap().pointer_input(len);

        let wgpu = self.wgpu_init.read().unwrap();

        // binding
        let heap_binding = &self.heap_binding;
        let array_a_binding = array_a.binding();
        let array_b_binding = array_b.binding();
        let out_binding = self.create_metadata_binding(
            &[allocate.1, allocate.2],
            &output_shape,
            &stride,
            &stride,
            &0,
        );

        // pipeline
        // // shader
        let shader = wgpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shader For Matmul 2D"),
            source: ShaderSource::Wgsl(include_str!("./matmul_nd.wgsl").into()),
        });

        // // pipeline_layout
        let pipeline_layout = wgpu.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Matmul ND"),
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
        let pipeline = wgpu.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline For Matmul ND"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                layout: Some(&pipeline_layout),
                module: &shader,
            }),
        );

        let mut encoder = wgpu.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("Create Encoder For Matmul 2D"),
            }),
        );

        {
            let mut bcp = encoder.begin_compute_pass(
                &(ComputePassDescriptor {
                    label: Some("Create Begin Compute Pass For Matmul 2D"),
                    timestamp_writes: None,
                }),
            );

            // set pipeline
            bcp.set_pipeline(&pipeline);

            // set bind group
            // // heap
            bcp.set_bind_group(0, Some(&heap_binding.binding_groups), &[]);
            // // array a
            bcp.set_bind_group(1, Some(&array_a_binding.1), &[]);
            // // array b
            bcp.set_bind_group(2, Some(&array_b_binding.1), &[]);
            // // out
            bcp.set_bind_group(3, Some(&out_binding.1), &[]);

            let x = (shape_a[shape_a.len() - 2] + 16 - 1) / 16;
            let y = (shape_b[shape_b.len() - 1] + 16 - 1) / 16;
            let z = output_shape[..output_shape.len() - 2]
                .iter()
                .product::<u32>();
            bcp.dispatch_workgroups(x, y, z);
        }
        wgpu.queue.submit(Some(encoder.finish()));
        // wgpu.device.poll(wgpu::wgt::PollType::Wait).unwrap();

        let array = GpuArray {
            module: Arc::new(self.clone()),
            pointer: (allocate.1, allocate.2),
            shape: output_shape,
            length: len as usize,
            stride,
            space_type: allocate.0,
            binding: out_binding,
        };

        Ok(array)
    }
}
