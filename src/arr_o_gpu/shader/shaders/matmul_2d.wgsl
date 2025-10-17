// heap
@group(0) @binding(0)
var <storage, read_write> heap: array<f32>;

// array A
@group(1) @binding(0)
var <storage, read> pointer_a: vec2<u32>;

@group(1) @binding(1)
var <storage, read> shape_a: vec2<u32>;

@group(1) @binding(2)
var <storage, read> stride_a: vec2<u32>;

// array B
@group(1) @binding(3)
var <storage, read> pointer_b: vec2<u32>;

@group(1) @binding(4)
var <storage, read> shape_b: vec2<u32>;

@group(1) @binding(5)
var <storage, read> stride_b: vec2<u32>;

// output
@group(1) @binding(6)
var <storage, read_write> output: array<f32>;

@group(1) @binding(7)
var <storage, read> stride_output: vec3<u32>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id:vec3<u32>){
    let m = global_id.x;
    let n = global_id.y;
    let k = global_id.z;

    if m < shape_a[0] && n < shape_b[1] && k < shape_b[0]{
        let index_of_a = index_2d(m, k, stride_a);
        let index_of_b = index_2d(k, n, stride_b);

        let value_a = heap[pointing(pointer_a, index_of_a)];
        let value_b = heap[pointing(pointer_b, index_of_b)];
        let result = value_a * value_b;
        
        let out_index = index_3d(m, n, k, stride_output);
        output[out_index] = result;
    }
}

fn pointing(pointer:vec2<u32>, index:u32) -> u32{
    return pointer.x + index;
}

fn index_2d(m: u32, n: u32, stride: vec2<u32>) -> u32{
    return stride.x * m + stride.y * n;
}

fn index_3d(m: u32, n: u32, k:u32, stride: vec3<u32>) -> u32{
    return stride.x * m + stride.y * n + stride.z * k;
}