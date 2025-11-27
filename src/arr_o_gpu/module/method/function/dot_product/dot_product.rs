use std::sync::{Arc, RwLockReadGuard};

use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BufferBindingType, BufferUsages, ComputePassDescriptor, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderStages,
    util::{BufferInitDescriptor, DeviceExt},
    wgt::CommandEncoderDescriptor,
};

use crate::{ArrOgpuErr, ArrOgpuModule, GpuArray, WgpuInit};

impl ArrOgpuModule {
    pub fn dot_product(
        &self,
        array_a: &GpuArray,
        array_b: &GpuArray,
    ) -> Result<GpuArray, ArrOgpuErr> {
        let shape_a = array_a.shape();
        let shape_b = array_b.shape();

        let length_a = array_a.len();
        let length_b = array_b.len();

        if shape_a.len() != 1 || shape_b.len() != 1 || length_a != length_b {
            let err = format!(
                "Array Dot Product Error, Array with shape {:?} and With Shape {:?} Can't Be Operated",
                shape_a, shape_b
            );
            return Err(ArrOgpuErr::DotProduct(err));
        }

        let length = length_a as u32;
        let pointer_a = array_a.pointer_to_arr();
        let pointer_b = array_b.pointer_to_arr();
        let output_allocate = self.allocator_write().pointer_input(1);

        let pointer_output = [output_allocate.1, output_allocate.2];

        let wgpu_init = self.wgpu_init().read().unwrap();

        // bind_group
        let heap_bind_group = &self.binding_compounds.read().unwrap()[0];
        let (bing_group, bind_group_layout) =
            bind_group_dot_product(&wgpu_init, length, &pointer_a, &pointer_b, &pointer_output);

        // pipeline and shader
        let pipeline_layout = wgpu_init
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Dot Product"),
                push_constant_ranges: &[],
                bind_group_layouts: &[
                    // heap
                    &heap_bind_group.binding_group_layouts,
                    &bind_group_layout,
                ],
            });

        let shader = wgpu_init
            .device
            .create_shader_module(ShaderModuleDescriptor {
                label: Some("Create Shaders For Dot Product"),
                source: wgpu::ShaderSource::Wgsl(include_str!("./dot_product.wgsl").into()),
            });

        let pipeline = wgpu_init
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Create Pipeline Layout For Dot Product"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                layout: Some(&pipeline_layout),
                module: &shader,
            });

        let mut encoder = wgpu_init
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Create Encoder For Dot Product"),
            });

        {
            let mut bcp = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Create Compute Pass For Dot Product"),
                timestamp_writes: None,
            });

            bcp.set_pipeline(&pipeline);

            // group 0
            bcp.set_bind_group(0, Some(&heap_bind_group.binding_groups), &[]);
            // group 1
            bcp.set_bind_group(1, Some(&bing_group), &[]);
            bcp.dispatch_workgroups(1, 1, 1);
        }

        wgpu_init.queue.submit(Some(encoder.finish()));
        wgpu_init.device.poll(wgpu::wgt::PollType::Wait).unwrap();

        let arr = GpuArray {
            length: length_a,
            module: Arc::new(self.clone()),
            pointer: (output_allocate.1 as usize, output_allocate.2 as usize),
            shape: vec![1],
            space_type: output_allocate.0,
            stride: vec![1],
        };

        Ok(arr)
    }
}

pub(crate) fn bind_group_dot_product(
    wgpu_init: &RwLockReadGuard<'_, WgpuInit>,
    k: u32,
    pointer_a: &[u32],
    pointer_b: &[u32],
    pointer_output: &[u32],
) -> (wgpu::BindGroup, wgpu::BindGroupLayout) {
    // Buffer
    // // k
    let k_buffer = wgpu_init.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Create k Buffer For Dot Product"),
        contents: bytemuck::bytes_of(&k),
        usage: BufferUsages::UNIFORM,
    });

    // array a
    // // pointer
    let pointer_a_buffer = wgpu_init.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Create Pointer A Buffer For Dot Product"),
        contents: bytemuck::cast_slice(pointer_a),
        usage: BufferUsages::UNIFORM,
    });

    // array b
    // // pointer
    let pointer_b_buffer = wgpu_init.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Create Pointer B Buffer For Dot Product"),
        contents: bytemuck::cast_slice(pointer_b),
        usage: BufferUsages::UNIFORM,
    });

    // output
    // // pointer
    let pointer_output_buffer = wgpu_init.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Create Pointer Output Buffer For Dot Product"),
        contents: bytemuck::cast_slice(pointer_output),
        usage: BufferUsages::UNIFORM,
    });

    // bind group layout
    let bind_group_layout = wgpu_init
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Create Binding Group Layout Of Output For Dot Product"),
            entries: &[
                // k (length)
                BindGroupLayoutEntry {
                    binding: 0,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    visibility: ShaderStages::COMPUTE,
                },
                // array a
                // // pointer
                BindGroupLayoutEntry {
                    binding: 1,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    visibility: ShaderStages::COMPUTE,
                },
                // array b
                // // pointer
                BindGroupLayoutEntry {
                    binding: 2,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    visibility: ShaderStages::COMPUTE,
                },
                // output
                // // pointer
                BindGroupLayoutEntry {
                    binding: 3,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    visibility: ShaderStages::COMPUTE,
                },
            ],
        });

    let bind_group = wgpu_init.device.create_bind_group(&BindGroupDescriptor {
        label: Some("Create Binding Group Layout Of Output For Dot Product"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: k_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: pointer_a_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: pointer_b_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: pointer_output_buffer.as_entire_binding(),
            },
        ],
    });

    (bind_group, bind_group_layout)
}
