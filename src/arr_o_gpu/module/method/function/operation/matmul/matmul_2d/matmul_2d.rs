use std::sync::Arc;

use wgpu::{
    ComputePassDescriptor,
    ComputePipelineDescriptor,
    PipelineCompilationOptions,
    PipelineLayoutDescriptor,
    ShaderModuleDescriptor,
    ShaderSource,
    wgt::CommandEncoderDescriptor,
};

use crate::{ ArrOgpuErr, ArrOgpuModule, ArrayView, GpuArray, get_stride_from_shape };

impl ArrOgpuModule {
    pub fn matmul_2d<'a, A, B>(&self, array_a: &A, array_b: &B) -> Result<GpuArray, ArrOgpuErr>
        where A: ArrayView, B: ArrayView
    {
        if array_a.dim() != 2 || array_b.dim() != 2 {
            let err = format!(
                "Array matmul 2d Error, dim of array A is {} and dim of array B is {}",
                array_a.dim(),
                array_b.dim()
            );
            return Err(ArrOgpuErr::Matmul2D(err));
        }

        let shape_a = array_a.shape();
        let shape_b = array_b.shape();

        let m_k = [shape_a[0], shape_a[1]];
        let k_n = [shape_b[0], shape_b[1]];
        if m_k[1] != k_n[0] {
            let err = format!(
                "Array matmul 2d Error, Array A {:?} can't matmul with Array B {:?}",
                shape_a,
                shape_b
            );
            return Err(ArrOgpuErr::Matmul2D(err));
        }

        let wgpu = self.wgpu_init.read().unwrap();
        let out_shape = [m_k[0], k_n[1]];
        let stride = get_stride_from_shape(&out_shape);
        let len = out_shape.iter().product::<u32>();
        let allocate = self.allocator.write().unwrap().pointer_input(len);

        // binding
        // // heap
        let heap_binding = &self.heap_binding;
        // // array a
        let array_a_binding = array_a.binding();
        // // array b
        let array_b_binding = array_b.binding();
        // // out
        let out_binding = self.array_data_binding(
            &[allocate.1, allocate.2],
            &out_shape,
            &stride,
            &stride,
            &0
        );

        // pipeline
        // // shader
        let shader = wgpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shader For Matmul 2D"),
            source: ShaderSource::Wgsl(include_str!("./matmul_2d.wgsl").into()),
        });

        // // pipeline_layout
        let pipeline_layout = wgpu.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Matmul 2D"),
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

        // // pipeline
        let pipeline = wgpu.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline For Matmul 2D"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                layout: Some(&pipeline_layout),
                module: &shader,
            })
        );

        // encoder
        let mut encoder = wgpu.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("Create Encoder For Matmul 2D"),
            })
        );

        {
            // begin compute pass
            let mut bcp = encoder.begin_compute_pass(
                &(ComputePassDescriptor {
                    label: Some("Create Begin Compute Pass For Matmul 2D"),
                    timestamp_writes: None,
                })
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

            // dispatch
            let x = (m_k[0] + 16 - 1) / 16;
            let y = (k_n[1] + 16 - 1) / 16;
            bcp.dispatch_workgroups(x, y, 1);
        }

        // submit
        wgpu.queue.submit(Some(encoder.finish()));

        // sync
        wgpu.device.poll(wgpu::wgt::PollType::Wait).unwrap();

        let array = GpuArray {
            module: Arc::new(self.clone()),
            length: len as usize,
            pointer: (allocate.1, allocate.2),
            shape: out_shape.to_vec(),
            stride,
            space_type: allocate.0,
            binding: out_binding,
        };

        Ok(array)
    }
}
