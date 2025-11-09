use std::sync::RwLockWriteGuard;

use wgpu::{
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayout,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BindingType,
    BufferBindingType,
    BufferUsages,
    Device,
    ShaderStages,
    util::{ BufferInitDescriptor, DeviceExt },
};

use crate::{ Allocator, GpuArray, get_stride_from_shape };

pub(crate) fn broadcast_bind_group(
    device: &Device,
    arr: &GpuArray,
    mut allocator: RwLockWriteGuard<'_, Allocator>,
    broadcast: &[u32],
    broadcast_target: usize
) -> (BindGroupLayout, wgpu::BindGroup, u32, u32, (crate::SpaceType, u32, u32)) {
    let shape = &arr.shape;
    let stride = &arr.stride;

    // thread_limit
    let thread_limit = shape[..broadcast_target].iter().product::<u32>();
    let thread_limit_buffer = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Thread Limit  Layout For BroadCast"),
            contents: bytemuck::bytes_of(&thread_limit),
            usage: BufferUsages::UNIFORM,
        })
    );

    // stride_target
    let stride_target = stride[broadcast_target];
    let stride_target_buffer = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Stride Target  Layout For BroadCast"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::bytes_of(&stride_target),
        })
    );

    // extend_count
    let extend_count = broadcast[broadcast_target];
    let extend_count_buffer = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Stride Target Layout For BroadCast"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::bytes_of(&extend_count),
        })
    );

    // output pointer & stride
    let output_len = broadcast.iter().product::<u32>();
    let output_allocate = allocator.pointer_input(output_len);
    let output_pointer = [output_allocate.1, output_allocate.2];
    let output_stride = get_stride_from_shape(&broadcast);
    let output_stride_target = output_stride[broadcast_target] * extend_count;

    let output_stride_target_buffer = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Output Stride Target Layout For BroadCast"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::bytes_of(&output_stride_target),
        })
    );

    // arr_pointer
    let arr_pointer = arr.pointer_to_arr();
    let arr_pointer_buffer = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Arr Pointer Layout For BroadCast"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::bytes_of(&arr_pointer),
        })
    );

    // out_pointer
    let out_pointer_buffer = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Output Pinter Layout For BroadCast"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::bytes_of(&output_pointer),
        })
    );

    let bind_group_layout = device.create_bind_group_layout(
        &(BindGroupLayoutDescriptor {
            label: Some(""),
            entries: &[
                // thread_limit
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
                // stride_target
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
                // stride_output
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
                // extend_count
                BindGroupLayoutEntry {
                    binding: 3,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                // arr_pointer
                BindGroupLayoutEntry {
                    binding: 4,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                // out_pointer
                BindGroupLayoutEntry {
                    binding: 5,
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
    let bind_group = device.create_bind_group(
        &(BindGroupDescriptor {
            label: Some("()"),
            layout: &bind_group_layout,
            entries: &[
                // thread_limit
                BindGroupEntry {
                    binding: 0,
                    resource: thread_limit_buffer.as_entire_binding(),
                },
                // stride_target
                BindGroupEntry {
                    binding: 1,
                    resource: stride_target_buffer.as_entire_binding(),
                },
                // stride_output
                BindGroupEntry {
                    binding: 2,
                    resource: output_stride_target_buffer.as_entire_binding(),
                },
                // extend_count
                BindGroupEntry {
                    binding: 3,
                    resource: extend_count_buffer.as_entire_binding(),
                },
                // arr_pointer
                BindGroupEntry {
                    binding: 4,
                    resource: arr_pointer_buffer.as_entire_binding(),
                },
                // out_pointer
                BindGroupEntry {
                    binding: 5,
                    resource: out_pointer_buffer.as_entire_binding(),
                },
            ],
        })
    );

    (bind_group_layout, bind_group, thread_limit, output_stride_target, output_allocate)
}

// @group(1) @binding(0)
// var<uniform> thread_limit: u32;

// @group(1) @binding(1)
// var<uniform> stride_target: u32;

// @group(1) @binding(2)
// var<uniform> stride_output: u32;

// @group(1) @binding(3)
// var<uniform> extend_count: u32;

// @group(1) @binding(4)
// var<uniform> arr_pointer: vec2<u32>;

// @group(1) @binding(5)
// var<uniform> out_pointer: vec2<u32>;
