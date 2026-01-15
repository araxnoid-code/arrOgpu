// mod sum;
mod sum_axis;

use std::{num::NonZero, sync::Arc, vec};

use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, Buffer, BufferUsages, CommandEncoder, ShaderStages,
    util::DeviceExt,
};

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, ArrayType, GpuArray, MetadataCompound,
    arr_o_gpu::compute_shaders::{SUM_CONTIGUOUS_SHADERS_PATH, SUM_VIEW_SHADERS_PATH},
};

impl ArrOgpuModule {
    pub fn sum<A>(&self, array_a: &A) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
    {
        let wgpu = self.wgpu_init().read().unwrap();
        // set up metadata
        let shape = vec![1];
        let stride = vec![1];
        let offset = 0;
        let dim = 1;
        let len = 1;
        let allocate = self.allocator.write().unwrap().pointer_input(len);

        let output_metadata = self.create_metadata_compound(
            [allocate.1, allocate.2],
            len,
            dim,
            offset,
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
        );

        // reduction
        // // reduction_len_list
        let mut reduction_len_list = vec![];
        let mut dummy = array_a.len();
        loop {
            dummy = (dummy + 511) >> 9;
            reduction_len_list.push(Counter::init(dummy));

            if dummy == 1 {
                break;
            }
        }

        let reduction_len_list_buffer =
            wgpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Create Reduction Len List Buffer"),
                    contents: bytemuck::cast_slice(&reduction_len_list),
                    usage: BufferUsages::UNIFORM,
                });

        // // reduction_counter
        let reduction_counter = [Counter::init(0), Counter::init(1)];
        let reduction_counter_buffer =
            wgpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Create Reduction Counter Buffer"),
                    contents: bytemuck::cast_slice(&reduction_counter),
                    usage: BufferUsages::UNIFORM,
                });

        // // reduction_heap
        let reduction_heap_buffer = wgpu.device.create_buffer(&wgpu::wgt::BufferDescriptor {
            label: Some("Create Reduction Heap Buffer"),
            size: reduction_len_list[0].value as u64 * 4,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            wgpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Create Bind Group Layout"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            visibility: ShaderStages::COMPUTE,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: true,
                                min_binding_size: None,
                            },
                            visibility: ShaderStages::COMPUTE,
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            count: None,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: true,
                                min_binding_size: None,
                            },
                            visibility: ShaderStages::COMPUTE,
                        },
                    ],
                });

        let bind_group = wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Create BindGroup For Sum"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: reduction_heap_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &reduction_len_list_buffer,
                        offset: 0,
                        size: NonZero::new(256),
                    }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &reduction_counter_buffer,
                        offset: 0,
                        size: NonZero::new(256),
                    }),
                },
            ],
        });

        let pipeline_layout = wgpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Create Pipeline For Sum"),
                bind_group_layouts: &[
                    &self.heap_binding().binding_group_layouts,
                    &bind_group_layout,
                ],
                immediate_size: 0,
            });

        let pipeline = wgpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Create Pipeline For Sum"),
                layout: Some(&pipeline_layout),
                module: &wgpu
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Create Pipeline For Sum"),
                        source: wgpu::ShaderSource::Wgsl(
                            match array_a.check_contiguous_or_view() {
                                ArrayType::Contiguous(_) => SUM_CONTIGUOUS_SHADERS_PATH,
                                _ => SUM_VIEW_SHADERS_PATH,
                            }
                            .into(),
                        ),
                    }),
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: true,
                },
                cache: None,
            });

        let mut encoder =
            wgpu.device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Create Encoder For Sum"),
                });

        set_execute_cache(&mut encoder, array_a, &self.execute_args, &output_metadata)?;

        for (i, len) in reduction_len_list.iter().enumerate() {
            let mut begin_compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Create Begin Compute Pass For Sum"),
                timestamp_writes: None,
            });

            begin_compute_pass.set_pipeline(&pipeline);
            begin_compute_pass.set_bind_group(0, Some(&self.heap_binding().binding_groups), &[]);

            begin_compute_pass.set_bind_group(
                1,
                Some(&bind_group),
                &[
                    (i as u32 - 1 * (i != 0) as u32) * 256,
                    (i != 0) as u32 * 256,
                ],
            );

            begin_compute_pass.dispatch_workgroups(len.value, 1, 1);

            if len.value == 1 {
                break;
            }
        }

        wgpu.queue.submit(Some(encoder.finish()));

        let array = GpuArray {
            shape,
            binding: None,
            length: len as usize,
            metadata_compound: Some(output_metadata),
            module: Arc::new(self.clone()),
            pointer: (allocate.1, allocate.2),
            space_type: allocate.0,
            stride,
        };

        Ok(array)
    }
}

fn set_execute_cache<A>(
    encoder: &mut CommandEncoder,
    array_a: &A,
    execute_cache: &Arc<Buffer>,
    output_metadata: &MetadataCompound,
) -> Result<(), ArrOgpuErr>
where
    A: ArrayCompute,
{
    let metadata_a = array_a.metadata_compound().ok_or(ArrOgpuErr::DotProduct(
        "Sum Error, Metadata Not Yet Defined For Array A".to_string(),
    ))?;

    let size = match array_a.check_contiguous_or_view() {
        ArrayType::Contiguous(_) => 32,
        _ => 256,
    };

    encoder.copy_buffer_to_buffer(&metadata_a.buffer, 0, execute_cache, size * 0, size);
    encoder.copy_buffer_to_buffer(&output_metadata.buffer, 0, execute_cache, size * 1, size);
    Ok(())
}

#[repr(C)]
#[derive(Pod, Zeroable, Copy, Clone, Debug)]
struct Counter {
    value: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
    padding3: [u32; 60],
}

impl Counter {
    fn init(value: u32) -> Counter {
        Self {
            value,
            padding0: 0,
            padding1: 0,
            padding2: 0,
            padding3: [0; 60],
        }
    }
}
