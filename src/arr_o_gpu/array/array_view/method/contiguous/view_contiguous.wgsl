// heap
@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;

// array A
// // pointer
@group(1) @binding(0)
var<uniform> pointer: vec2<u32>;

// // shape
@group(1) @binding(1)
var<storage, read> shape: array<u32>;

// // iters
@group(1) @binding(2)
var<storage, read> iters: array<u32>;

// // stride
@group(1) @binding(3)
var<storage, read> stride: array<u32>;

// // offset
@group(1) @binding(4)
var<uniform> offset: u32;


// Output
// // pointer
@group(2) @binding(0)
var<uniform> pointer_out: vec2<u32>;

// // shape
@group(2) @binding(1)
var<storage, read> shape_out: array<u32>;

// // iters
@group(2) @binding(2)
var<storage, read> iters_out: array<u32>;

// // stride
@group(2) @binding(3)
var<storage, read> stride_out: array<u32>;

// // offset
@group(2) @binding(4)
var<uniform> offset_out: u32;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id:vec3<u32>){
    let len = pointer_out.y - pointer_out.x;
    heap[20] = f32(1000000000000);
    if (global_id.x < len){
        let shape_len = arrayLength(&shape);
        var idx = offset + pointer.x;
        let idx_out = global_id.x + pointer_out.x;

        for (var dim = 0u; dim < shape_len; dim++){
            let iter = iters[dim];
            let range = shape[dim];
            let permute = (global_id.x / iter) % range;
            idx += permute * stride[dim];
        }
        heap[idx_out] = heap[idx];
    }
}