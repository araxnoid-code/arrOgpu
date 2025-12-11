// heap
@group(0) @binding(0)
var<storage, read_write> heap:array<f32>;

// array A
// // pointer
@group(1) @binding(0)
var<uniform> pointer_a: vec2<u32>;

// // shape
@group(1) @binding(1)
var<storage, read> shape_a: array<u32>;

// // iters
@group(1) @binding(2)
var<storage, read> iters_a: array<u32>;

// // stride
@group(1) @binding(3)
var<storage, read> stride_a: array<u32>;

// // offset
@group(1) @binding(4)
var<uniform> offset_a: u32;

// array B
// // pointer
@group(2) @binding(0)
var<uniform> pointer_b: vec2<u32>;

// // shape
@group(2) @binding(1)
var<storage, read> shape_b: array<u32>;

// // iters
@group(2) @binding(2)
var<storage, read> iters_b: array<u32>;

// // stride
@group(2) @binding(3)
var<storage, read> stride_b: array<u32>;

// // offset
@group(2) @binding(4)
var<uniform> offset_b: u32;

// output
// // pointer
@group(3) @binding(0)
var<uniform> pointer_o: vec2<u32>;

// // shape
@group(3) @binding(1)
var<storage, read> shape_o: array<u32>;

// // iters
@group(3) @binding(2)
var<storage, read> iters_o: array<u32>;

// // stride
@group(3) @binding(3)
var<storage, read> stride_o: array<u32>;

// // offset
@group(3) @binding(4)
var<uniform> offset_o: u32;


@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id:vec3<u32>){
    let length = pointer_o.y - pointer_o.x;
    if (global_id.x < length){
        let idx_a = indexing_array_a(global_id.x) + pointer_a.x;
        let idx_b = indexing_array_b(global_id.x) + pointer_b.x;
        let idx_o = global_id.x + pointer_o.x;

        heap[idx_o] = heap[idx_a] - heap[idx_b];
    }

}

fn indexing_array_a(i:u32) -> u32{
    let len = arrayLength(&shape_a);
    var index = offset_a;
    for (var dim = 0u; dim < len; dim+=1){
        let iter = iters_a[dim];
        let permute = (i / iter) % shape_a[dim];
        index += permute * stride_a[dim];
    }
    return index;
}

fn indexing_array_b(i:u32) -> u32{
    let len = arrayLength(&shape_b);
    var index = offset_b;
    for (var dim = 0u; dim < len; dim+=1){
        let iter = iters_b[dim];
        let permute = (i / iter) % shape_b[dim];
        index += permute * stride_b[dim];
    }
    return index;
}