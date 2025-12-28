use std::{num::NonZero, sync::Arc, vec};

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, BindingResource, BindingType, BufferUsages, ShaderStages,
};

use crate::{ArrOgpuErr, ArrOgpuModule, ArrayCompute, ArrayType, CheckArrayType, GpuArray};

impl ArrOgpuModule {
    pub fn dot_product_optimize<'a, A, B>(
        &self,
        array_a: &'a A,
        array_b: &'a B,
    ) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute + CheckArrayType<'a>,
        B: ArrayCompute + CheckArrayType<'a>,
    {
        let shape_a = array_a.shape();
        let shape_b = array_b.shape();

        // error hendling
        if let ArrayType::View(_) = array_a.check() {
            let err = format!("Dot Product Error, dot_product_optimize Not Implemented View Yet");
            return Err(ArrOgpuErr::DotProduct(err));
        } else if let ArrayType::View(_) = array_b.check() {
            let err = format!("Dot Product Error, dot_product_optimize Not Implemented View Yet");
            return Err(ArrOgpuErr::DotProduct(err));
        } else if shape_a.len() != 1 && shape_b.len() != 1 {
            let err = format!(
                "Dot Product Error, Array A with shape {:?} and Array B With Shape {:?} Can't Be Operated",
                shape_a, shape_b
            );

            return Err(ArrOgpuErr::DotProduct(err));
        } else if array_a.len() != array_b.len() {
            let err = format!(
                "Dot Product Error, Array A And Array B Do Not Have The Same Length So They Cannot Be Operated On",
            );

            return Err(ArrOgpuErr::DotProduct(err));
        }

        let wgpu = self.wgpu_init().read().unwrap();

        // output metadata
        let len = 1;
        let shape = vec![1];
        let stride = vec![1];
        let allocate = self.allocator_write().pointer_input(len);

        // bind
        let heap_bind = self.heap_binding();
        let array_a_bind = array_a.binding();
        let array_b_bind = array_b.binding();
        let out_bind =
            self.create_metadata_binding(&[allocate.1, allocate.2], &shape, &stride, &stride, &0);

        // reduction
        // // reduction_counter
        let reduction_counter = create_reduction_counter();
        let window_reduction_counter = wgpu.device.create_buffer(&wgpu::wgt::BufferDescriptor {
            label: Some("Create Window Reduction Counter For Dot Product"),
            size: 256 * 2,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // // reduction_len
        let reduction_len = create_reduction_len(array_a.len());

        let window_reduction_len = wgpu.device.create_buffer(&wgpu::wgt::BufferDescriptor {
            label: Some("Create Reduction For Dot Product"),
            size: (reduction_len.len() * 256) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // // reduction
        let out_len = (array_a.len() + 511) / 512;
        let reduction_buffer = wgpu.device.create_buffer(&wgpu::wgt::BufferDescriptor {
            label: Some("Create Reduction For Dot Product"),
            size: (out_len * 4) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // reduction bind
        let reduction_bind_layout =
            wgpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Create Bind Group Layout Reduction For Dot Product"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            count: None,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            count: None,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: true,
                                min_binding_size: None,
                            },
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            count: None,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: true,
                                min_binding_size: None,
                            },
                        },
                    ],
                });

        let reduction_bind = wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Create Bind Group Reduction For Dot Produt"),
            layout: &reduction_bind_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: reduction_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &window_reduction_counter,
                        offset: 0,
                        size: Some(NonZero::new(256).unwrap()),
                    }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &window_reduction_len,
                        offset: 0,
                        size: Some(NonZero::new(256).unwrap()),
                    }),
                },
            ],
        });

        let pipeline_layout = wgpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Dot Product"),
                bind_group_layouts: &[
                    &heap_bind.binding_group_layouts,
                    &array_a_bind.0,
                    &array_b_bind.0,
                    &out_bind.0,
                    &reduction_bind_layout,
                ],
                push_constant_ranges: &[],
            });

        let shader = wgpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Create Shaders Module For Dot Product"),
                source: wgpu::ShaderSource::Wgsl(include_str!("./dot_product_op.wgsl").into()),
            });

        let pipeline = wgpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Create Pipeline For Dot Product"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        wgpu.queue.write_buffer(
            &window_reduction_counter,
            0,
            bytemuck::cast_slice(&reduction_counter),
        );

        wgpu.queue.write_buffer(
            &window_reduction_len,
            0,
            bytemuck::cast_slice(&reduction_len),
        );

        let mut encoder =
            wgpu.device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Create Command Encoder For Dot Porduct"),
                });

        let mut x = out_len;
        let offset_metadata = 256;
        let mut counter = 0;
        let mut len_counter = 0;
        loop {
            let mut bcp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Create Begin Compute Pass For Dot Product"),
                timestamp_writes: None,
            });

            bcp.set_pipeline(&pipeline);
            bcp.set_bind_group(0, Some(&heap_bind.binding_groups), &[]);
            bcp.set_bind_group(1, Some(&array_a_bind.1), &[]);
            bcp.set_bind_group(2, Some(&array_a_bind.1), &[]);
            bcp.set_bind_group(3, Some(&out_bind.1), &[]);

            bcp.set_bind_group(
                4,
                Some(&reduction_bind),
                &[counter * offset_metadata, len_counter * offset_metadata],
            );

            bcp.dispatch_workgroups(x, 1, 1);

            if counter > 0 {
                len_counter += 1;
            }

            if counter == 0 {
                counter += 1;
            }

            if x == 1 {
                break;
            } else {
                x = (x + 511) / 512;
            }
        }

        wgpu.queue.submit(Some(encoder.finish()));

        let array = GpuArray {
            module: Arc::new(self.clone()),
            length: len as usize,
            binding: out_bind,
            pointer: (allocate.1, allocate.2),
            shape,
            stride,
            space_type: allocate.0,
        };

        Ok(array)
    }
}

fn create_reduction_counter() -> [Counter; 2] {
    [
        Counter {
            value: 0,
            padding: [0; 63],
        },
        Counter {
            value: 1,
            padding: [0; 63],
        },
    ]
}

fn create_reduction_len(len: u32) -> Vec<Counter> {
    let mut reduction_len = vec![];
    let mut len = len;
    loop {
        let out_len = (len + 511) / 512;

        if out_len == 1 {
            reduction_len.push(Counter {
                value: 1,
                padding: [0; 63],
            });
            break;
        } else {
            len = out_len;
            reduction_len.push(Counter {
                value: len,
                padding: [0; 63],
            });
        }
    }

    reduction_len
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct Counter {
    value: u32,
    padding: [u32; 63],
}
