use std::sync::Arc;

use wgpu::{
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BindingType,
    BufferBindingType,
    BufferUsages,
    CommandEncoderDescriptor,
    ComputePipelineDescriptor,
    PipelineCompilationOptions,
    PipelineLayoutDescriptor,
    ShaderModuleDescriptor,
    ShaderStages,
    wgt::BufferDescriptor,
};

use crate::{ ArrOgpuErr, ArrOgpuModule, ArrayView, GpuArray };

impl ArrOgpuModule {
    pub fn sum<A>(&self, array: &A) -> Result<GpuArray, ArrOgpuErr> where A: ArrayView {
        let wgpu = self.wgpu_init.read().unwrap();
        // out meta data
        let shape = vec![1];
        let stride = vec![1];
        let len = 1;
        let allocate = self.allocator.write().unwrap().pointer_input(len);

        // bind group
        let heap_bind = &*self.heap_binding;
        let array_bind = array.binding();
        let out_bind = self.array_data_binding(
            &[allocate.1, allocate.2],
            &shape,
            &stride,
            &stride,
            &0
        );

        let reduction_bind_group_layout = wgpu.device.create_bind_group_layout(
            &(BindGroupLayoutDescriptor {
                label: Some("Create Reduction Result Bind Group Layout"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        count: None,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        visibility: ShaderStages::COMPUTE,
                    },
                ],
            })
        );

        let mut x = ((array.len() as u32) + 512 - 1) / 512;
        let size = (x + 1) * (std::mem::size_of::<f32>() as u32);
        let reduction_res_buffer = wgpu.device.create_buffer(
            &(BufferDescriptor {
                label: Some("Create Reduction Result Buffer For Sum"),
                mapped_at_creation: false,
                usage: BufferUsages::STORAGE,
                size: size as u64,
            })
        );

        let reduction_bind_group = wgpu.device.create_bind_group(
            &(BindGroupDescriptor {
                label: Some("Create Reduction Result Bind Group"),
                layout: &reduction_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: reduction_res_buffer.as_entire_binding(),
                    },
                ],
            })
        );

        // pipeline
        let shader = wgpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shader Module For Sum"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./sum_optimaze.wgsl").into()),
        });

        let pipeline = wgpu.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline For Sum"),
                bind_group_layouts: &[
                    &heap_bind.binding_group_layouts,
                    &array_bind.0,
                    &out_bind.0,
                    &reduction_bind_group_layout,
                ],
                push_constant_ranges: &[],
            })
        );

        let pipeline = wgpu.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline For Sum"),
                layout: Some(&pipeline),
                module: &shader,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                cache: None,
            })
        );

        loop {
            // encoder
            let mut encoder = wgpu.device.create_command_encoder(
                &(CommandEncoderDescriptor {
                    label: Some("Create Encoder For Sum"),
                })
            );

            {
                let mut bcp = encoder.begin_compute_pass(
                    &(wgpu::ComputePassDescriptor {
                        label: Some("Create Begin Compute Pass For Sum"),
                        timestamp_writes: None,
                    })
                );

                bcp.set_pipeline(&pipeline);

                bcp.set_bind_group(0, Some(&heap_bind.binding_groups), &[]);
                bcp.set_bind_group(1, Some(&array_bind.1), &[]);
                bcp.set_bind_group(2, Some(&out_bind.1), &[]);
                bcp.set_bind_group(3, Some(&reduction_bind_group), &[]);

                bcp.dispatch_workgroups(x, 1, 1);
            }

            wgpu.queue.submit(Some(encoder.finish()));

            if x > 1 {
                x = (x + 512 - 1) / 512;
            } else {
                break;
            }
        }

        // if let Err(err_poll) = wgpu.device.poll(wgpu::wgt::PollType::Wait) {
        // let error = "Add Error, Error While Poll".to_string();
        // return Err(ArrOgpuErr::Poll(error, err_poll));
        // }

        let array = GpuArray {
            module: Arc::new(self.clone()),
            binding: out_bind,
            length: len as usize,
            pointer: (allocate.1, allocate.2),
            shape,
            space_type: allocate.0,
            stride: stride,
        };

        Ok(array)
    }
}
