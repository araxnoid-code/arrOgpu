use std::sync::{ Arc, RwLock };

use wgpu::{ BufferUsages, Device, util::{ BufferInitDescriptor, DeviceExt } };

use crate::{ Allocator, GpuArray, get_stride_from_shape };

pub(crate) fn matmul_nd_group_binding(
    device: &Device,
    allocator: &Arc<RwLock<Allocator>>,
    arr_a: &GpuArray,
    arr_b: &GpuArray
) {
    // A
    // Pointer A
    let pointer_a = arr_a.pointer_to_arr();
    let buffer_pointer_a = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Pointer A Buffer Layout For Matmul ND"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::cast_slice(&pointer_a),
        })
    );

    // matrix stride of A
    let len_of_stride = arr_a.stride.len();
    let matrix_stride_a = &arr_a.stride[len_of_stride - 2..];
    let buffer_matrix_stride_a = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Matrix Stride A Buffer Layout For Matmul ND"),
            contents: bytemuck::cast_slice(matrix_stride_a),
            usage: BufferUsages::UNIFORM,
        })
    );

    // shape of A
    let shape_len = arr_a.shape.len();
    let shape_of_matrix_a = &arr_a.shape[shape_len - 2..];
    let buffer_shape_of_matrix_a = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Matrix Shape A Buffer Layout For Matmul ND"),
            contents: bytemuck::cast_slice(shape_of_matrix_a),
            usage: BufferUsages::UNIFORM,
        })
    );

    // B
    // Pointer B
    let pointer_b = arr_b.pointer_to_arr();
    let buffer_pointer_b = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Pointer B Buffer Layout For Matmul ND"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::cast_slice(&pointer_b),
        })
    );

    // matrix stride of B
    let len_of_stride = arr_b.stride.len();
    let matrix_stride_b = &arr_b.stride[len_of_stride - 2..];
    let buffer_matrix_stride_b = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Matrix Stride B Buffer Layout For Matmul ND"),
            contents: bytemuck::cast_slice(matrix_stride_b),
            usage: BufferUsages::UNIFORM,
        })
    );

    // shape of B
    let shape_len = arr_b.shape.len();
    let shape_of_matrix_b = &arr_b.shape[shape_len - 2..];
    let buffer_shape_of_matrix_b = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Matrix Shape B Buffer Layout For Matmul ND"),
            contents: bytemuck::cast_slice(shape_of_matrix_b),
            usage: BufferUsages::UNIFORM,
        })
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
        })
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
        })
    );

    // other
    // stride_between_matrix
    let stride_between_matrix = arr_a.stride[arr_a.stride.len() - 3];
    let buffer_stride_between_matrix = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create stride between matrix Buffer Layout For Matmul ND"),
            contents: bytemuck::bytes_of(&stride_between_matrix),
            usage: BufferUsages::UNIFORM,
        })
    );

    // stride_between_matrix_out
    let stride_between_matrix_out = stride_out[stride_out.len() - 3];
    let buffer_stride_between_matrix_out = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create stride between matrix Buffer Layout For Matmul ND"),
            contents: bytemuck::bytes_of(&stride_between_matrix_out),
            usage: BufferUsages::UNIFORM,
        })
    );
}

// // heap
// @group(0) @binding(0)
// var <storage, read_write> heap: array<f32>;

// // A
// // pointer
// @group(1) @binding(0)
// var <uniform> pointer_a: vec2<u32>;

// // stride of matrix
// @group(1) @binding(1)
// var <uniform> matrix_stride_a: vec2<u32>;

// // shape of matrix
// @group(1) @binding(2)
// var <uniform> matrix_shape_a: vec2<u32>;

// // B
// // pointer
// @group(1) @binding(3)
// var <uniform> pointer_b: vec2<u32>;

// // stride of matrix
// @group(1) @binding(4)
// var <uniform> matrix_stride_b: vec2<u32>;

// // shape of matrix
// @group(1) @binding(5)
// var <uniform> matrix_shape_b: vec2<u32>;

// // output
// // pointer ouput
// @group(1) @binding(6)
// var <uniform> pointer_out: vec2<u32>;

// // stride of matrix
// @group(1) @binding(7)
// var <uniform> matrix_stride_out: vec2<u32>;

// // other
// @group(1) @binding(8)
// var <uniform> stride_between_matrix: u32; // = length of matrix

// @group(1) @binding(9)
// var <uniform> stride_between_matrix_out: u32; // = length of output matrix
