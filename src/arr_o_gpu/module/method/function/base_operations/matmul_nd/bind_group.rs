use std::sync::{Arc, RwLock};

use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, BufferUsages, Device, ShaderStages,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::{Allocator, GpuArray, get_stride_from_shape};

pub(crate) fn matmul_nd_group_binding(
    device: &Device,
    allocator: &Arc<RwLock<Allocator>>,
    arr_a: &GpuArray,
    arr_b: &GpuArray,
) -> (
    wgpu::BindGroupLayout,
    wgpu::BindGroup,
    Vec<u32>,
    Vec<u32>,
    (crate::SpaceType, u32, u32),
) {
    // A
    // Pointer A
    let pointer_a = arr_a.pointer_to_arr();
    let buffer_pointer_a = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Pointer A Buffer Layout For Matmul ND"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::cast_slice(&pointer_a),
        }),
    );

    // matrix stride of A
    let len_of_stride = arr_a.stride.len();
    let matrix_stride_a = &arr_a.stride[len_of_stride - 2..];
    let buffer_matrix_stride_a = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Matrix Stride A Buffer Layout For Matmul ND"),
            contents: bytemuck::cast_slice(matrix_stride_a),
            usage: BufferUsages::UNIFORM,
        }),
    );

    // shape of A
    let shape_len = arr_a.shape.len();
    let shape_of_matrix_a = &arr_a.shape[shape_len - 2..];
    let buffer_shape_of_matrix_a = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Matrix Shape A Buffer Layout For Matmul ND"),
            contents: bytemuck::cast_slice(shape_of_matrix_a),
            usage: BufferUsages::UNIFORM,
        }),
    );

    // B
    // Pointer B
    let pointer_b = arr_b.pointer_to_arr();
    let buffer_pointer_b = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Pointer B Buffer Layout For Matmul ND"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::cast_slice(&pointer_b),
        }),
    );

    // matrix stride of B
    let len_of_stride = arr_b.stride.len();
    let matrix_stride_b = &arr_b.stride[len_of_stride - 2..];
    let buffer_matrix_stride_b = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Matrix Stride B Buffer Layout For Matmul ND"),
            contents: bytemuck::cast_slice(matrix_stride_b),
            usage: BufferUsages::UNIFORM,
        }),
    );

    // shape of B
    let shape_len = arr_b.shape.len();
    let shape_of_matrix_b = &arr_b.shape[shape_len - 2..];
    let buffer_shape_of_matrix_b = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Matrix Shape B Buffer Layout For Matmul ND"),
            contents: bytemuck::cast_slice(shape_of_matrix_b),
            usage: BufferUsages::UNIFORM,
        }),
    );

    // Output
    // pointer output
    let mut output_shape = arr_a.shape.clone();
    *output_shape.last_mut().unwrap() = *arr_b.shape.last().unwrap();
    let out_len = output_shape.iter().product::<u32>();
    let allocate_out = allocator.write().unwrap().pointer_input(out_len);
    let pointer_output = [allocate_out.1, allocate_out.2];
    let buffer_pointer_output = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Pointer Output Buffer Layout For Matmul ND"),
            contents: bytemuck::cast_slice(&pointer_output),
            usage: BufferUsages::UNIFORM,
        }),
    );

    // matrix stride of pointer
    let stride_out = get_stride_from_shape(&output_shape);
    let len = stride_out.len();
    let matrix_stride_out = &stride_out[len - 2..];
    let buffer_matrix_stride_out = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Matrix Stride Output Buffer Layout For Matmul ND"),
            contents: bytemuck::cast_slice(matrix_stride_out),
            usage: BufferUsages::UNIFORM,
        }),
    );

    // other
    // stride_between_matrix_a
    let stride_between_matrix_a = arr_a.stride[arr_a.stride.len() - 3];
    let buffer_stride_between_matrix_a = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create stride between matrix Buffer Layout For Matmul ND"),
            contents: bytemuck::bytes_of(&stride_between_matrix_a),
            usage: BufferUsages::UNIFORM,
        }),
    );

    // stride_between_matrix_b
    let stride_between_matrix_b = arr_b.stride[arr_a.stride.len() - 3];
    let buffer_stride_between_matrix_b = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create stride between matrix Buffer Layout For Matmul ND"),
            contents: bytemuck::bytes_of(&stride_between_matrix_b),
            usage: BufferUsages::UNIFORM,
        }),
    );

    // stride_between_matrix_out
    let stride_between_matrix_out = stride_out[stride_out.len() - 3];
    let buffer_stride_between_matrix_out = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create stride between matrix Buffer Layout For Matmul ND"),
            contents: bytemuck::bytes_of(&stride_between_matrix_out),
            usage: BufferUsages::UNIFORM,
        }),
    );

    let bind_group_layout = bind_group_layout(device);
    let bind_group = device.create_bind_group(
        &(BindGroupDescriptor {
            label: Some("Create Bind Group For Matmul ND"),
            layout: &bind_group_layout,
            entries: &[
                // A
                // pointer A
                BindGroupEntry {
                    binding: 0,
                    resource: buffer_pointer_a.as_entire_binding(),
                },
                // matrix stride of A
                BindGroupEntry {
                    binding: 1,
                    resource: buffer_matrix_stride_a.as_entire_binding(),
                },
                // shape of A
                BindGroupEntry {
                    binding: 2,
                    resource: buffer_shape_of_matrix_a.as_entire_binding(),
                },
                // B
                // pointer B
                BindGroupEntry {
                    binding: 3,
                    resource: buffer_pointer_b.as_entire_binding(),
                },
                // matrix stride of B
                BindGroupEntry {
                    binding: 4,
                    resource: buffer_matrix_stride_b.as_entire_binding(),
                },
                // shape of B
                BindGroupEntry {
                    binding: 5,
                    resource: buffer_shape_of_matrix_b.as_entire_binding(),
                },
                // Output
                // pointer output
                BindGroupEntry {
                    binding: 6,
                    resource: buffer_pointer_output.as_entire_binding(),
                },
                // matrix stride of pointer
                BindGroupEntry {
                    binding: 7,
                    resource: buffer_matrix_stride_out.as_entire_binding(),
                },
                // other
                // stride_between_matrix_a
                BindGroupEntry {
                    binding: 8,
                    resource: buffer_stride_between_matrix_a.as_entire_binding(),
                },
                // stride_between_matrix_b
                BindGroupEntry {
                    binding: 9,
                    resource: buffer_stride_between_matrix_b.as_entire_binding(),
                },
                // stride_between_matrix_out
                BindGroupEntry {
                    binding: 10,
                    resource: buffer_stride_between_matrix_out.as_entire_binding(),
                },
            ],
        }),
    );

    (
        bind_group_layout,
        bind_group,
        output_shape,
        stride_out,
        allocate_out,
    )
}

fn bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
    let bind_group_layout = device.create_bind_group_layout(
        &(BindGroupLayoutDescriptor {
            label: Some("Create Bind Group Layout For Matmul ND"),
            entries: &[
                // A
                // // pointer
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
                // // stride of matrix
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
                // // shape of matrix
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
                // B
                // // pointer
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
                // // stride of matrix
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
                // // shape of matrix
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
                // output
                // // pointer ouput
                BindGroupLayoutEntry {
                    binding: 6,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                // // stride of matrix
                BindGroupLayoutEntry {
                    binding: 7,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                // other
                BindGroupLayoutEntry {
                    binding: 8,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 9,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 10,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ],
        }),
    );

    bind_group_layout
}
