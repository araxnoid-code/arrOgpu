use std::sync::RwLockReadGuard;

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

use crate::WgpuInit;

pub(crate) fn bind_group_collect(
    wgpu: &RwLockReadGuard<'_, WgpuInit>,
    pointer: &[u32],
    shape: &[u32],
    stride: &[u32],
    offset: &u32,
    product: &u32,
    pointer_out: &[u32]
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    // array
    // // pointer
    let pointer_buffer = wgpu.device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Pointer Buffer Layout For Collect View"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::cast_slice(pointer),
        })
    );

    // // shape
    let shape_buffer = wgpu.device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Shape Buffer Layout For Collect View"),
            usage: BufferUsages::STORAGE,
            contents: bytemuck::cast_slice(shape),
        })
    );

    // // stride
    let stride_buffer = wgpu.device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Stride Buffer Layout For Collect View"),
            usage: BufferUsages::STORAGE,
            contents: bytemuck::cast_slice(stride),
        })
    );

    // // offset
    let offset_buffer = wgpu.device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Offset Buffer Layout For Collect View"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::bytes_of(offset),
        })
    );

    // // product
    let len_buffer = wgpu.device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Len / Product Buffer Layout For Collect View"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::bytes_of(product),
        })
    );

    // output
    // // pointer
    let pointer_out_buffer = wgpu.device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Output Pointer Buffer Layout For Collect View"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::cast_slice(pointer_out),
        })
    );

    let bind_group_layout = wgpu.device.create_bind_group_layout(
        &(BindGroupLayoutDescriptor {
            label: Some("Create Bind Group Layout Of Output For Collect View"),
            entries: &[
                // array
                // // pointer
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
                // // shape
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
                // // stride
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
                // // offset
                BindGroupLayoutEntry {
                    binding: 3,
                    count: None,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    visibility: ShaderStages::COMPUTE,
                },
                // // product
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
                // // pointer_out
                BindGroupLayoutEntry {
                    binding: 5,
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
            label: Some("Create Bind Group Layout Of Output For Collect View"),
            layout: &bind_group_layout,
            entries: &[
                // array
                // // pointer
                BindGroupEntry {
                    binding: 0,
                    resource: pointer_buffer.as_entire_binding(),
                },
                // // shape
                BindGroupEntry {
                    binding: 1,
                    resource: shape_buffer.as_entire_binding(),
                },
                // // stride
                BindGroupEntry {
                    binding: 2,
                    resource: stride_buffer.as_entire_binding(),
                },
                // // offset
                BindGroupEntry {
                    binding: 3,
                    resource: offset_buffer.as_entire_binding(),
                },
                // // product
                BindGroupEntry {
                    binding: 4,
                    resource: len_buffer.as_entire_binding(),
                },
                // // pointer_out
                BindGroupEntry {
                    binding: 5,
                    resource: pointer_out_buffer.as_entire_binding(),
                },
            ],
        })
    );

    (bind_group_layout, bind_group)
}
