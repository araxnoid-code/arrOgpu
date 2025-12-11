use std::sync::RwLockReadGuard;

use wgpu::{
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BufferBindingType,
    BufferUsages,
    ShaderStages,
    util::{ BufferInitDescriptor, DeviceExt },
};

use crate::WgpuInit;

pub(crate) fn bind_group_dot_product(
    wgpu_init: &RwLockReadGuard<'_, WgpuInit>,
    k: u32,
    pointer_a: &[u32],
    pointer_b: &[u32],
    pointer_output: &[u32]
) -> (wgpu::BindGroup, wgpu::BindGroupLayout) {
    // Buffer
    // // k
    let k_buffer = wgpu_init.device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create k Buffer For Dot Product"),
            contents: bytemuck::bytes_of(&k),
            usage: BufferUsages::UNIFORM,
        })
    );

    // array a
    // // pointer
    let pointer_a_buffer = wgpu_init.device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Pointer A Buffer For Dot Product"),
            contents: bytemuck::cast_slice(pointer_a),
            usage: BufferUsages::UNIFORM,
        })
    );

    // array b
    // // pointer
    let pointer_b_buffer = wgpu_init.device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Pointer B Buffer For Dot Product"),
            contents: bytemuck::cast_slice(pointer_b),
            usage: BufferUsages::UNIFORM,
        })
    );

    // output
    // // pointer
    let pointer_output_buffer = wgpu_init.device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Pointer Output Buffer For Dot Product"),
            contents: bytemuck::cast_slice(pointer_output),
            usage: BufferUsages::UNIFORM,
        })
    );

    // bind group layout
    let bind_group_layout = wgpu_init.device.create_bind_group_layout(
        &(BindGroupLayoutDescriptor {
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
        })
    );

    let bind_group = wgpu_init.device.create_bind_group(
        &(BindGroupDescriptor {
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
        })
    );

    (bind_group, bind_group_layout)
}
