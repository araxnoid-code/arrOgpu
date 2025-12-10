// heap
@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;

// array
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

// // product
@group(1) @binding(5)
var<uniform> len: u32;

// output
// // pointer
@group(1) @binding(6)
var<uniform> pointer_out: vec2<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id:vec3<u32>){
    if (global_id.x < len){
        let shape_len = arrayLength(&shape);
        var idx = offset;
        for(var i = 0u; i < shape_len; i++){
            let permute = ( global_id.x / iters[i] ) % shape[i];
            idx += permute * stride[i];
        }

        let array_index = idx + pointer.x;
        let output_index = global_id.x + pointer_out.x;

        heap[output_index] = heap[array_index];
    }
}