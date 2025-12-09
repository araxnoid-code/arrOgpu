use std::sync::Arc;

use wgpu::{
    util::{ BufferInitDescriptor, DeviceExt },
    wgt::{ CommandEncoderDescriptor, PollType },
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
    ShaderStages,
};

use crate::{ get_stride_from_shape, ArrOgpuErr, ArrOgpuModule, GpuArray };

impl ArrOgpuModule {
    pub fn sub(&self, arr_a: &GpuArray, arr_b: &GpuArray) -> Result<GpuArray, ArrOgpuErr> {
        let wgpu_init = self.wgpu_init.read().unwrap();
        let len = arr_a.pointer.1 - arr_a.pointer.0;

        if arr_a.shape != arr_b.shape {
            let err = format!(
                "Array Sub Error, Shape Of A is {:?} but adding with Shape Of B is {:?}",
                arr_a.shape,
                arr_b.shape
            );
            return Err(ArrOgpuErr::Sub(err));
        }

        // source
        let pointer_a = arr_a.pointer_to_arr();
        let pointer_a = wgpu_init.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some("Create Pointer A Buffer Layout For Sub"),
                usage: BufferUsages::UNIFORM,
                contents: bytemuck::cast_slice(&pointer_a),
            })
        );

        let pointer_b = arr_b.pointer_to_arr();
        let pointer_b = wgpu_init.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some("Create Pointer A Buffer Layout For Sub"),
                usage: BufferUsages::UNIFORM,
                contents: bytemuck::cast_slice(&pointer_b),
            })
        );

        let allocate = self.allocator
            .write()
            .unwrap()
            .pointer_input(len as u32);
        let pointer_type = allocate.0;
        let pointer_out = [allocate.1, allocate.2];
        let pointer_out = wgpu_init.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some("Create Pointer A Buffer Layout For Sub"),
                usage: BufferUsages::UNIFORM,
                contents: bytemuck::cast_slice(&pointer_out),
            })
        );

        // binding
        let binding_layout = wgpu_init.device.create_bind_group_layout(
            &(BindGroupLayoutDescriptor {
                label: Some("Create Binding Group Layout Of Output For Sub"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        count: None,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        count: None,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        count: None,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                ],
            })
        );

        let binding = wgpu_init.device.create_bind_group(
            &(BindGroupDescriptor {
                label: Some("Create Binding Group Layout Of Output For Sub"),
                layout: &binding_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: pointer_a.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: pointer_b.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: pointer_out.as_entire_binding(),
                    },
                ],
            })
        );

        // pipeline
        let shader = wgpu_init.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shaders For Sub"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./sub.wgsl").into()),
        });
        let binding_of_heap = &self.binding_compounds.read().unwrap()[0];
        let pipeline_layout = wgpu_init.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Sub"),
                bind_group_layouts: &[&binding_of_heap.binding_group_layouts, &binding_layout],
                push_constant_ranges: &[],
            })
        );

        let pipeline = wgpu_init.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline For Sub"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("main"),
                layout: Some(&pipeline_layout),
                module: &shader,
            })
        );

        let mut encoder = wgpu_init.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("Create Encoder For Sub"),
            })
        );

        // bgein compute pas

        {
            let mut bcp = encoder.begin_compute_pass(
                &(ComputePassDescriptor {
                    label: Some("Create Compute Pass For Sub"),
                    timestamp_writes: None,
                })
            );

            bcp.set_pipeline(&pipeline);
            bcp.set_bind_group(0, Some(&binding_of_heap.binding_groups), &[]);
            bcp.set_bind_group(1, Some(&binding), &[]);
            let x = ((len as f32) / 256.0).ceil() as u32;
            bcp.dispatch_workgroups(x, 1, 1);
        }

        wgpu_init.queue.submit(Some(encoder.finish()));
        wgpu_init.device.poll(PollType::Wait).unwrap();

        let arr = GpuArray {
            length: len as usize,
            module: Arc::new(self.clone()),
            pointer: (allocate.1, allocate.2),
            shape: arr_a.shape.clone(),
            space_type: pointer_type,
            stride: get_stride_from_shape(&arr_a.shape),
            binding: None,
        };

        Ok(arr)
    }
}
