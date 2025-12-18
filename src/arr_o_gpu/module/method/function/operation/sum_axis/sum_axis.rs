use std::sync::Arc;

use wgpu::{
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayout,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BindingType,
    BufferBindingType,
    BufferUsages,
    ComputePassDescriptor,
    ComputePipelineDescriptor,
    Device,
    PipelineCompilationOptions,
    PipelineLayoutDescriptor,
    ShaderModuleDescriptor,
    ShaderStages,
    util::{ BufferInitDescriptor, DeviceExt },
    wgt::CommandEncoderDescriptor,
};

use crate::{
    ArrOgpuErr,
    ArrOgpuModule,
    ArrayView,
    GpuArray,
    arr_o_gpu::module::method::function::operation::sum_axis::error_handling::{ error_handling },
    get_stride_from_shape,
};

impl ArrOgpuModule {
    pub fn sum_axis<A>(&self, array_a: &A, axis: &[u32]) -> Result<GpuArray, ArrOgpuErr>
        where A: ArrayView
    {
        let mut axis = axis.to_vec();
        axis.sort();

        // error handling
        error_handling(array_a, &axis)?;

        // out meta data
        let array_shape = array_a.shape();
        let mut out_shape = array_shape.clone();

        if array_shape.len() != axis.len() {
            axis.iter()
                .rev()
                .for_each(|idx| {
                    out_shape.remove(*idx as usize);
                });
        } else {
            out_shape = vec![1];
        }

        let stride = get_stride_from_shape(&out_shape);
        let out_len = out_shape.iter().product::<u32>();
        let allocate = self.allocator.write().unwrap().pointer_input(out_len);

        // binding
        let heap_binding = &self.heap_binding;
        let array_binding = array_a.binding();
        let out_binding = self.array_data_binding(
            &[allocate.1, allocate.2],
            &out_shape,
            &stride,
            &stride,
            &0
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
                    &array_binding.0,
                    &out_binding.0,
                    &others_binding.0,
                ],
                push_constant_ranges: &[],
            })
        );

        let pipeline = wgpu.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline For Sum Axis"),
                layout: Some(&pipeline),
                module: &shader,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                cache: None,
            })
        );

        // encoder
        let mut encoder = wgpu.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("Create Encoder For Sum Axis"),
            })
        );

        {
            // begin compute pass
            let mut bcp = encoder.begin_compute_pass(
                &(ComputePassDescriptor {
                    label: Some("Create Begin Compute Pass For Sum Axis"),
                    timestamp_writes: None,
                })
            );

            // set pipeline
            bcp.set_pipeline(&pipeline);

            // bind group
            // // heap
            bcp.set_bind_group(0, Some(&heap_binding.binding_groups), &[]);

            // // array
            bcp.set_bind_group(1, Some(&array_binding.1), &[]);

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
        wgpu.device.poll(wgpu::wgt::PollType::Wait).unwrap();

        let array = GpuArray {
            module: Arc::new(self.clone()),
            binding: out_binding,
            length: out_len as usize,
            pointer: (allocate.1, allocate.2),
            space_type: allocate.0,
            shape: out_shape,
            stride: stride,
        };

        Ok(array)
    }
}

fn others_binding(
    device: &Device,
    axis_list: &[u32],
    shape_of_slice: &[u32],
    stride_of_slice: &[u32]
) -> (BindGroupLayout, wgpu::BindGroup) {
    // buffer
    // // axis_list
    let axis_list_buffer = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Axis List Buffer For Init For Sum Axis"),
            contents: bytemuck::cast_slice(axis_list),
            usage: BufferUsages::STORAGE,
        })
    );

    // // shape_of_slice
    let shape_of_slice_buffer = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Axis List Buffer For Init For Sum Axis"),
            contents: bytemuck::cast_slice(shape_of_slice),
            usage: BufferUsages::STORAGE,
        })
    );

    // // stride_of_slice
    let stride_of_slice_buffer = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Axis List Buffer For Init For Sum Axis"),
            contents: bytemuck::cast_slice(stride_of_slice),
            usage: BufferUsages::STORAGE,
        })
    );

    // binding layout
    let binding_layout = device.create_bind_group_layout(
        &(BindGroupLayoutDescriptor {
            label: Some("Create Others Binding Layout For Sum Axis"),
            entries: &[
                // axis_list
                BindGroupLayoutEntry {
                    binding: 0,
                    count: None,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    visibility: ShaderStages::COMPUTE,
                },

                // shape_of_slice
                BindGroupLayoutEntry {
                    binding: 1,
                    count: None,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    visibility: ShaderStages::COMPUTE,
                },

                // stride_of_slice
                BindGroupLayoutEntry {
                    binding: 2,
                    count: None,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    visibility: ShaderStages::COMPUTE,
                },
            ],
        })
    );

    let binding = device.create_bind_group(
        &(BindGroupDescriptor {
            label: Some("Create Others Binding For Sum Axis"),
            layout: &binding_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: axis_list_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: shape_of_slice_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: stride_of_slice_buffer.as_entire_binding(),
                },
            ],
        })
    );

    (binding_layout, binding)
}
