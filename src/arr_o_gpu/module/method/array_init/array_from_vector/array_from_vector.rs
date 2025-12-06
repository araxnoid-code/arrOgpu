use std::{ sync::{ Arc, RwLock }, vec };

use wgpu::{
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BindingType,
    BufferBindingType,
    BufferUsages,
    ComputePassDescriptor,
    ComputePipelineDescriptor,
    PipelineCompilationOptions,
    PipelineLayoutDescriptor,
    ShaderModuleDescriptor,
    ShaderSource,
    ShaderStages,
    util::{ BufferInitDescriptor, DeviceExt },
    wgt::{ CommandEncoderDescriptor, PollType },
};

use crate::*;

impl ArrOgpuModule {
    // array init
    pub fn array_from_vector(&self, vector: &[f32], shape: &[u32]) -> Result<GpuArray, ArrOgpuErr> {
        let (flatten, _) = vector.flatten();

        let len = flatten.len();
        let shape_len = shape.iter().product::<u32>();

        if (len as u32) != shape_len {
            return Err(
                ArrOgpuErr::Init(
                    "Array Initialization Error, len of vector and len of shape not same".to_string()
                )
            );
        }

        let wgpu = &self.wgpu_init.read().unwrap();
        let vector_buffer = wgpu.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some(&format!("create array buffer with shape:{shape:?}")),
                contents: bytemuck::cast_slice(&flatten),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            })
        );

        // allocator
        let pointer = self.allocator
            .write()
            .unwrap()
            .pointer_input(len as u32);
        let space_type = pointer.0;
        let pointer = [pointer.1, pointer.2];

        // pointer
        let pointer_buffer = wgpu.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some(&format!("create buffer pointer array with pointing:{pointer:?}")),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
                contents: bytemuck::cast_slice(&pointer),
            })
        );

        // group 1
        let binding_layout = wgpu.device.create_bind_group_layout(
            &(BindGroupLayoutDescriptor {
                label: Some("Create array bind group layout for array_init"),
                entries: &[
                    // buffer
                    BindGroupLayoutEntry {
                        binding: 0,
                        count: None,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                    },
                    // pointer
                    BindGroupLayoutEntry {
                        binding: 1,
                        count: None,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            min_binding_size: None,
                            has_dynamic_offset: false,
                        },
                    },
                ],
            })
        );

        // group 1
        let binding = wgpu.device.create_bind_group(
            &(BindGroupDescriptor {
                label: Some("Create array bind layout for array_init"),
                layout: &binding_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: vector_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: pointer_buffer.as_entire_binding(),
                    },
                ],
            })
        );

        let pipeline_layout = wgpu.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("create pipeline layout for array_init"),
                bind_group_layouts: &[
                    &self.binding_compounds.read().unwrap()[0].binding_group_layouts, // heap
                    &binding_layout,
                ],
                push_constant_ranges: &[],
            })
        );

        let shaders = wgpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("create shaders module 'array_init.wgsl'"),
            source: ShaderSource::Wgsl(include_str!("./array_init.wgsl").into()),
        });
        let pipeline = wgpu.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("create pipeline for array_init"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("array_init"),
                layout: Some(&pipeline_layout),
                module: &shaders,
            })
        );

        let mut encoder = wgpu.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("encoder for array_init"),
            })
        );

        {
            let mut bcp = encoder.begin_compute_pass(
                &(ComputePassDescriptor {
                    label: Some("create init compute pass descriptor for array_init"),
                    timestamp_writes: None,
                })
            );

            bcp.set_pipeline(&pipeline);

            // heap
            bcp.set_bind_group(0, &self.binding_compounds.read().unwrap()[0].binding_groups, &[]);

            bcp.set_bind_group(1, &binding, &[]);
            let x = ((len as f32) / 256.0).ceil() as u32;
            bcp.dispatch_workgroups(x, 1, 1);
        }

        wgpu.queue.submit(Some(encoder.finish()));

        wgpu.device.poll(PollType::Wait).unwrap();

        let pointer = (pointer[0], pointer[1]);

        Ok(GpuArray {
            module: Arc::new(self.clone()),
            pointer,
            length: (pointer.1 - pointer.0) as usize,
            shape: shape.to_vec(),
            stride: get_stride_from_shape(shape),
            space_type: space_type,
        })
    }
}
