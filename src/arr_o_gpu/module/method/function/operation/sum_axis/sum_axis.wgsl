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

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
){
    // get slice representation
    let x = global_id.x;
    let length_output = pointer_o.y - pointer_o.x;
}