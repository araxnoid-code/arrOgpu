use std::{num::NonZero, sync::Arc};

use bytemuck::{Pod, Zeroable, cast_slice};
use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, BindingType, BufferUsages, ShaderStages, util::DeviceExt,
};

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, GpuArray,
    arr_o_gpu::{
        compute_shaders::SUM_AXIS_SHADERS_PATH,
        module::method::function::operation::sum_axis::tools::error_handling,
    },
    get_stride_from_shape, vector_padding,
};

impl ArrOgpuModule {
    pub fn sum_axis<A>(&self, array: &A, axis: &[u32]) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
    {
        let mut axis = axis.to_vec();
        axis.sort();

        error_handling(array, &axis)?;
        let wgpu = self.wgpu_init().read().unwrap();

        let array_shape = array.shape();
        let mut out_shape = array_shape.clone();
        if array_shape.len() != axis.len() {
            axis.iter().rev().for_each(|idx| {
                out_shape.remove(*idx as usize);
            });
        } else {
            out_shape = vec![1];
        }

        let dim = out_shape.len();
        let stride = get_stride_from_shape(&out_shape);
        let out_len = out_shape.iter().product::<u32>();
        let allocate = self.allocator.write().unwrap().pointer_input(out_len);

        let shape_padding: [u32; 8] = vector_padding(out_shape.clone(), 0, 8)
            .map_err(|err| ArrOgpuErr::SumAxis(err))?
            .try_into()
            .unwrap();

        let stride_padding: [u32; 8] = vector_padding(stride.clone(), 0, 8)
            .map_err(|err| ArrOgpuErr::SumAxis(err))?
            .try_into()
            .unwrap();

        let metada_output = self.create_metadata_compound(
            [allocate.1, allocate.2],
            out_len,
            dim as u32,
            0,
            shape_padding,
            stride_padding,
            stride_padding,
        );

        let sum_len = axis
            .iter()
            .map(|axis| array.shape()[*axis as usize])
            .product::<u32>();

        let (pipeline, bind_group, reduction_len) =
            create_pipeline_and_reduction(&self, sum_len, out_len);

        let mut encoder =
            wgpu.device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Create Encoder For Sum Axis"),
                });

        let execute_args_buffer = &self.execute_args;

        let array_a_metadata = array.metadata_compound().unwrap();
        encoder.copy_buffer_to_buffer(&array_a_metadata.buffer, 0, execute_args_buffer, 0, 256);
        encoder.copy_buffer_to_buffer(&metada_output.buffer, 0, execute_args_buffer, 256, 256);

        let axis_padding =
            vector_padding(axis.clone(), 0, 8).map_err(|err| ArrOgpuErr::SumAxis(err))?;
        wgpu.queue
            .write_buffer(&self.static_cache, 0, cast_slice(&axis_padding));

        let x = (out_len + 15) >> 4;
        for (i, counter) in reduction_len.iter().enumerate() {
            let mut begin_compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Create Begin Compute Pass For Sum Axos"),
                timestamp_writes: None,
            });

            begin_compute_pass.set_pipeline(&pipeline);

            begin_compute_pass.set_bind_group(0, Some(&self.heap_binding().binding_groups), &[]);
            begin_compute_pass.set_bind_group(
                1,
                Some(&bind_group),
                &[
                    (i != 0) as u32 * 256,
                    (i as u32 - 1 * (i != 0) as u32) * 256,
                ],
            );

            begin_compute_pass.dispatch_workgroups(x, counter.value, 1);
        }

        wgpu.queue.submit(Some(encoder.finish()));

        let array = GpuArray {
            length: out_len as usize,
            metadata_compound: Some(metada_output),
            module: Arc::new(self.clone()),
            pointer: (allocate.1, allocate.2),
            shape: out_shape,
            space_type: allocate.0,
            stride,
        };

        Ok(array)
    }
}

fn create_pipeline_and_reduction(
    module: &ArrOgpuModule,
    len: u32,
    out_len: u32,
) -> (wgpu::ComputePipeline, wgpu::BindGroup, Vec<Counter>) {
    let device = &module.wgpu_init().read().unwrap().device;

    let reduction_counter = [Counter::init(0), Counter::init(1)];

    let mut reduction_len_list = vec![];
    let mut dummy = len;
    loop {
        dummy = (dummy + 15) >> 4;
        reduction_len_list.push(Counter::init(dummy));

        if dummy == 1 {
            break;
        }
    }

    let reduction_counter = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Create Reduction Counter Buffer For Sum Axis"),
        contents: cast_slice(&reduction_counter),
        usage: BufferUsages::UNIFORM,
    });

    let reduction_len = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Create Reduction Len Buffer For Sum Axis"),
        contents: cast_slice(&reduction_len_list),
        usage: BufferUsages::UNIFORM,
    });

    let reduction_heap = device.create_buffer(&wgpu::wgt::BufferDescriptor {
        label: Some("Create Reduction Heap Buffer For Sum Axis"),
        size: (reduction_len_list[0].value * out_len * 4) as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Create Bind Group Layout For Sum Axis"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                count: None,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                visibility: ShaderStages::COMPUTE,
            },
            BindGroupLayoutEntry {
                binding: 1,
                count: None,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                visibility: ShaderStages::COMPUTE,
            },
            BindGroupLayoutEntry {
                binding: 2,
                count: None,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                visibility: ShaderStages::COMPUTE,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Create Bind Group Layout For Sum Axis"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: reduction_heap.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &reduction_counter,
                    offset: 0,
                    size: NonZero::new(256),
                }),
            },
            BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &reduction_len,
                    offset: 0,
                    size: NonZero::new(256),
                }),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Create Pipeline Layout For Sum Axis"),
        bind_group_layouts: &[
            &module.heap_binding().binding_group_layouts,
            &bind_group_layout,
        ],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Create Pipeline For Sum Axis"),
        layout: Some(&pipeline_layout),
        module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Create Shaders Module For Sum Axis"),
            source: wgpu::ShaderSource::Wgsl(SUM_AXIS_SHADERS_PATH.into()),
        }),
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[],
            zero_initialize_workgroup_memory: true,
        },
        cache: None,
    });

    (pipeline, bind_group, reduction_len_list)
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
