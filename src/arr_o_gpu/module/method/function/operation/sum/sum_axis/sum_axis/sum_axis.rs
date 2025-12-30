use std::sync::Arc;

use wgpu::{
    ComputePassDescriptor, ComputePipelineDescriptor, PipelineCompilationOptions,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, wgt::CommandEncoderDescriptor,
};

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, GpuArray,
    arr_o_gpu::module::method::function::operation::sum::sum_axis::tools::{
        error_handling, others_binding,
    },
    get_stride_from_shape,
};

impl ArrOgpuModule {
    pub fn sum_axis<A>(&self, array_a: &A, axis: &[u32]) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
    {
        let mut axis = axis.to_vec();
        axis.sort();

        // error handling
        error_handling(array_a, &axis)?;

        // out meta data
        let array_shape = array_a.shape();
        let mut out_shape = array_shape.clone();

        if array_shape.len() != axis.len() {
            axis.iter().rev().for_each(|idx| {
                out_shape.remove(*idx as usize);
            });
        } else {
            out_shape = vec![1];
        }

        let stride = get_stride_from_shape(&out_shape);
        let out_len = out_shape.iter().product::<u32>();
        let allocate = self.allocator.write().unwrap().pointer_input(out_len);

        // binding
        let heap_binding = &self.module_bind_group;
        let array_binding = array_a.binding();
        let out_binding = self.create_metadata_binding(
            &[allocate.1, allocate.2],
            &out_shape,
            &stride,
            &stride,
            &0,
        );

        // wgpu
        let wgpu = self.wgpu_init.read().unwrap();

        // others
        let mut shape_of_slice = vec![1; array_shape.len()];
        for idx in &axis {
            shape_of_slice[*idx as usize] = array_shape[*idx as usize];
        }
        let stride_of_slice = get_stride_from_shape(&shape_of_slice);

        let others_binding = others_binding(&wgpu.device, &axis, &shape_of_slice, &stride_of_slice);

        // pipeline
        let shader = wgpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Created Shader Module For Sum Axis"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sum_axis.wgsl").into()),
        });

        let pipeline = wgpu.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline For Sum Axis"),
                bind_group_layouts: &[
                    &heap_binding.binding_group_layouts,
                    &array_binding.ok_or(ArrOgpuErr::refactor_err_0_1_0_5())?.0,
                    &out_binding.0,
                    &others_binding.0,
                ],
                immediate_size: 0,
            }),
        );

        let pipeline = wgpu.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline For Sum Axis"),
                layout: Some(&pipeline),
                module: &shader,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                cache: None,
            }),
        );

        // encoder
        let mut encoder = wgpu.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("Create Encoder For Sum Axis"),
            }),
        );

        {
            // begin compute pass
            let mut bcp = encoder.begin_compute_pass(
                &(ComputePassDescriptor {
                    label: Some("Create Begin Compute Pass For Sum Axis"),
                    timestamp_writes: None,
                }),
            );

            // set pipeline
            bcp.set_pipeline(&pipeline);

            // bind group
            // // heap
            bcp.set_bind_group(0, Some(&heap_binding.binding_groups), &[]);

            // // array
            bcp.set_bind_group(
                1,
                Some(&array_binding.ok_or(ArrOgpuErr::refactor_err_0_1_0_5())?.1),
                &[],
            );

            // // out
            bcp.set_bind_group(2, Some(&out_binding.1), &[]);

            // // others
            bcp.set_bind_group(3, Some(&others_binding.1), &[]);

            let x = (out_len + 16 - 1) / 16;
            let y = 1;
            let z = 1;
            bcp.dispatch_workgroups(x, y, z);
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
            binding: Some(out_binding),
            length: out_len as usize,
            pointer: (allocate.1, allocate.2),
            space_type: allocate.0,
            shape: out_shape,
            stride: stride,
        };

        Ok(array)
    }
}
