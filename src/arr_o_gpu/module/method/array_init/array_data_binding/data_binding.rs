use wgpu::{
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BindingType,
    BufferBindingType,
    BufferUsages,
    ShaderStages,
    util::{ BufferInitDescriptor, DeviceExt },
};

use crate::ArrOgpuModule;

impl ArrOgpuModule {
    pub fn array_data_binding(
        &self,
        pointer: &[u32],
        shape: &[u32],
        iters: &[u32],
        stride: &[u32],
        offset: &u32
    ) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        let wgpu = self.wgpu_init.read().unwrap();

        // pointer
        let pointer_buffer = wgpu.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some(""),
                usage: BufferUsages::UNIFORM,
                contents: bytemuck::cast_slice(pointer),
            })
        );

        // shape
        let shape_buffer = wgpu.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some(""),
                usage: BufferUsages::STORAGE,
                contents: bytemuck::cast_slice(shape),
            })
        );

        // iters
        let iters_buffer = wgpu.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some(""),
                usage: BufferUsages::STORAGE,
                contents: bytemuck::cast_slice(iters),
            })
        );

        // stride
        let stride_buffer = wgpu.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some(""),
                usage: BufferUsages::STORAGE,
                contents: bytemuck::cast_slice(stride),
            })
        );

        let offset_buffer = wgpu.device.create_buffer_init(
            &(BufferInitDescriptor {
                label: Some(""),
                usage: BufferUsages::UNIFORM,
                contents: bytemuck::bytes_of(offset),
            })
        );

        let bind_group_layout = wgpu.device.create_bind_group_layout(
            &(BindGroupLayoutDescriptor {
                label: Some("Create Bind Group Layout For Meta Data Of Array"),
                entries: &[
                    // pointer
                    BindGroupLayoutEntry {
                        binding: 0,
                        count: None,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        visibility: ShaderStages::COMPUTE,
                    },

                    // shape
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

                    // iters
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

                    // stride
                    BindGroupLayoutEntry {
                        binding: 3,
                        count: None,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        visibility: ShaderStages::COMPUTE,
                    },

                    // stride
                    BindGroupLayoutEntry {
                        binding: 4,
                        count: None,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        visibility: ShaderStages::COMPUTE,
                    },
                ],
            })
        );

        let bind_group = wgpu.device.create_bind_group(
            &(BindGroupDescriptor {
                label: Some(""),
                layout: &bind_group_layout,
                entries: &[
                    // pointer
                    BindGroupEntry {
                        binding: 0,
                        resource: pointer_buffer.as_entire_binding(),
                    },

                    // shape
                    BindGroupEntry {
                        binding: 1,
                        resource: shape_buffer.as_entire_binding(),
                    },

                    // iters
                    BindGroupEntry {
                        binding: 2,
                        resource: iters_buffer.as_entire_binding(),
                    },

                    // stride
                    BindGroupEntry {
                        binding: 3,
                        resource: stride_buffer.as_entire_binding(),
                    },

                    // offset
                    BindGroupEntry {
                        binding: 4,
                        resource: offset_buffer.as_entire_binding(),
                    },
                ],
            })
        );

        (bind_group_layout, bind_group)
    }
}
