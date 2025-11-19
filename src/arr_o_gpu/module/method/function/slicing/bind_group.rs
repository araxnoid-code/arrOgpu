use std::sync::RwLockReadGuard;

use wgpu::{
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayout,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BindingType,
    BufferBindingType,
    BufferUsages,
    ShaderStages,
    util::{ BufferInitDescriptor, DeviceExt },
};

use crate::WgpuInit;

pub(crate) fn bind_group_slicing(
    wgpu_init: &RwLockReadGuard<'_, WgpuInit>,
    start_slice: &Vec<u32>,
    end_slice: &Vec<u32>,
    _loop: &Vec<u32>,
    array_stride: &Vec<u32>,
    array_pointer: &[u32],
    output_pointer: &[u32]
) -> (BindGroupLayout, wgpu::BindGroup) {
    //  Buffer
    // // start_slice
    let start_slice_buffer = wgpu_init.device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Start Slicing  Layout For BroadCast"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::cast_slice(start_slice),
        })
    );

    // // end_slice
    let end_slice_buffer = wgpu_init.device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create End Slicing  Layout For BroadCast"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::cast_slice(end_slice),
        })
    );

    // // _loops
    let _loop_buffer = wgpu_init.device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create loop Layout For BroadCast"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::cast_slice(_loop),
        })
    );

    // // array
    // // // stride
    let array_stride_buffer = wgpu_init.device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Array Stride  Layout For BroadCast"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::cast_slice(array_stride),
        })
    );

    // // // pointer
    let array_pointer_buffer = wgpu_init.device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Array Pointer  Layout For BroadCast"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::cast_slice(array_pointer),
        })
    );

    // output
    // // pointer
    let output_pointer_buffer = wgpu_init.device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Output Pointer Layout For BroadCast"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::cast_slice(output_pointer),
        })
    );

    // binding layout
    let bind_group_layout = bind_group_layout(&wgpu_init);

    // binding
    let bind_group = wgpu_init.device.create_bind_group(
        &(BindGroupDescriptor {
            label: Some("Create Binding Group Layout Of Output For Slicing"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: start_slice_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: end_slice_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: _loop_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: array_stride_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: array_pointer_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: output_pointer_buffer.as_entire_binding(),
                },
            ],
        })
    );

    (bind_group_layout, bind_group)
}

fn bind_group_layout(wgpu_init: &RwLockReadGuard<'_, WgpuInit>) -> wgpu::BindGroupLayout {
    let binding_layout = wgpu_init.device.create_bind_group_layout(
        &(BindGroupLayoutDescriptor {
            label: Some("Create Binding Group Layout Of Output For Slicing"),
            entries: &[
                // slicing list
                // // slicing_list_start
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
                // // slicing_list_end
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
                // loop
                // // _loop
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
                // array
                // // stride
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
                // // pointer
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
                // output
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

    binding_layout
}

// // heap
// @group(0) @binding(0)
// var<storage, read_write> heap: array<f32>;

// // slicing list
// @group(1) @binding(0)
// var<uniform> slicing_list_start: array<u32>;
// @group(1) @binding(1)
// var<uniform> slicing_list_end: array<u32>;

// // loop
// @group(1) @binding(2)
// var<uniform> _loops: array<f32>;

// // stride
// @group(1) @binding(3)
// var<uniform> array_stride: array<u32>;

// // array pointer
// @group(1) @binding(4)
// var<uniform> array_pointer: vec2<u32>;

// // output pointer
// @group(1) @binding(5)
// var<uniform> output_pointer: vec2<u32>;
