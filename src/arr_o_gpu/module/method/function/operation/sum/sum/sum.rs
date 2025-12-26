use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferSize, BufferUsages, CommandEncoderDescriptor,
    ComputePipelineDescriptor, PipelineCompilationOptions, PipelineLayoutDescriptor,
    ShaderModuleDescriptor, ShaderStages, wgt::BufferDescriptor,
};

use crate::{ArrOgpuErr, ArrOgpuModule, ArrayCompute, ArrayType, CheckArrayType, GpuArray};

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod, Debug)]
struct ReducCounterMetaData {
    count: u32,
    padding: [u32; 63],
}

impl ArrOgpuModule {
    pub fn sum<'a, A>(&self, array: &'a A) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute + CheckArrayType<'a>,
    {
        let wgpu = self.wgpu_init.read().unwrap();
        // out meta data
        let shape = vec![1];
        let stride = vec![1];
        let len = 1;
        let allocate = self.allocator.write().unwrap().pointer_input(len);

        // bind group
        let heap_bind = &*self.heap_binding;
        let array_bind = array.binding();
        let out_bind =
            self.create_metadata_binding(&[allocate.1, allocate.2], &shape, &stride, &stride, &0);

        // reduction
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
                    BindGroupLayoutEntry {
                        binding: 1,
                        count: None,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: true,
                            min_binding_size: None,
                        },
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        count: None,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: true,
                            min_binding_size: None,
                        },
                    },
                ],
            }),
        );

        // list of output len
        let mut list_out_len = vec![];
        let mut len: u32 = array.len();
        loop {
            let new_len: u32 = (len + 511) / 512;

            if new_len == 1 {
                let out_len = ReducCounterMetaData {
                    count: 1,
                    padding: [0; 63],
                };
                list_out_len.push(out_len);
                break;
            } else {
                len = new_len;
                let out_len = ReducCounterMetaData {
                    count: len,
                    padding: [0; 63],
                };
                list_out_len.push(out_len);
            }
        }

        let list_out_len_buffer_window = wgpu.device.create_buffer(&BufferDescriptor {
            label: Some("Create List Out Len Buffer Window For Sum"),
            size: list_out_len.len() as u64 * 256,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut x = ((array.len() as u32) + 511) / 512;
        let size = x * (std::mem::size_of::<f32>() as u32);
        let reduction_res_buffer = wgpu.device.create_buffer(
            &(BufferDescriptor {
                label: Some("Create Reduction Result Buffer For Sum"),
                mapped_at_creation: false,
                usage: BufferUsages::STORAGE,
                size: size as u64,
            }),
        );

        let reduc_counter: Vec<ReducCounterMetaData> = vec![
            ReducCounterMetaData {
                count: 0,
                padding: [0; 63],
            },
            ReducCounterMetaData {
                count: 1,
                padding: [0; 63],
            },
        ];

        let reduc_counter_window_buffer = wgpu.device.create_buffer(&BufferDescriptor {
            label: Some("Create Reduc Counter Window Buffer For Sum"),
            size: 256 * 2,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let reduction_bind_group = wgpu.device.create_bind_group(
            &(BindGroupDescriptor {
                label: Some("Create Reduction Result Bind Group"),
                layout: &reduction_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: reduction_res_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &reduc_counter_window_buffer,
                            offset: 0,
                            size: BufferSize::new(256),
                        }),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &list_out_len_buffer_window,
                            offset: 0,
                            size: BufferSize::new(256),
                        }),
                    },
                ],
            }),
        );

        let shader = match array.check() {
            ArrayType::Contiguous(_) => wgpu.device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Create Shader Module For Sum"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("./shaders/sum_contiguous.wgsl").into(),
                ),
            }),
            ArrayType::View(_) => wgpu.device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Create Shader Module For Sum"),
                source: wgpu::ShaderSource::Wgsl(include_str!("./shaders/sum_view.wgsl").into()),
            }),
        };

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
            }),
        );

        let pipeline = wgpu.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline For Sum"),
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
                label: Some("Create Encoder For Sum"),
            }),
        );

        wgpu.queue.write_buffer(
            &reduc_counter_window_buffer,
            0,
            bytemuck::cast_slice(&reduc_counter),
        );

        wgpu.queue.write_buffer(
            &list_out_len_buffer_window,
            0,
            bytemuck::cast_slice(&list_out_len),
        );

        let offsetmetadata = 256;
        let mut counter = 0;
        let mut idx = 0;
        loop {
            {
                let mut bcp = encoder.begin_compute_pass(
                    &(wgpu::ComputePassDescriptor {
                        label: Some("Create Begin Compute Pass For Sum"),
                        timestamp_writes: None,
                    }),
                );

                bcp.set_pipeline(&pipeline);

                bcp.set_bind_group(0, Some(&heap_bind.binding_groups), &[]);
                bcp.set_bind_group(1, Some(&array_bind.1), &[]);
                bcp.set_bind_group(2, Some(&out_bind.1), &[]);
                bcp.set_bind_group(
                    3,
                    Some(&reduction_bind_group),
                    &[counter * offsetmetadata, offsetmetadata * idx],
                );

                bcp.dispatch_workgroups(x, 1, 1);
            }

            if counter > 0 {
                idx += 1;
            }

            if counter == 0 {
                counter += 1;
            }

            if x > 1 {
                x = (x + 511) / 512;
            } else {
                break;
            }
        }

        wgpu.queue.submit(Some(encoder.finish()));

        if let Err(err_poll) = wgpu.device.poll(wgpu::wgt::PollType::Wait) {
            let error = "Add Error, Error While Poll".to_string();
            return Err(ArrOgpuErr::Poll(error, err_poll));
        }

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
