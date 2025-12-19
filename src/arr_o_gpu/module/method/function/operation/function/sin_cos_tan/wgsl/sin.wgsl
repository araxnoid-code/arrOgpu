// heap
@group(0) @binding(0)
var<storage, read_write> heap:array<f32>;

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

// output
// // pointer
@group(2) @binding(0)
var<uniform> pointer_o: vec2<u32>;

// // shape
@group(2) @binding(1)
var<storage, read> shape_o: array<u32>;

// // iters
@group(2) @binding(2)
var<storage, read> iters_o: array<u32>;

// // stride
@group(2) @binding(3)
var<storage, read> stride_o: array<u32>;

// // offset
@group(2) @binding(4)
var<uniform> offset_o: u32;


@compute @workgroup_size(256, 1, 1)
fn main(@builtin (global_invocation_id) global_id:vec3<u32>){
    let length = pointer_o.y - pointer_o.x;
    if global_id.x < length{
        let index = indexing(global_id.x) + pointer.x;
        heap[pointer_o.x + global_id.x] = sin(heap[index]);
    }
}

fn indexing(x:u32) -> u32{
    var index = offset;
    let length = arrayLength(&shape);
    for (var i = 0u; i < length; i++){
        let permute = (x / iters[i]) % shape[i];
        index += (stride[i] * permute);
    }
    return index;
}