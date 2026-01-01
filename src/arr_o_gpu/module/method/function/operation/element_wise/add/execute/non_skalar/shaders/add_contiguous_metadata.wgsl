// override
override LEN: u32;
override START_POINTER_A: u32;
override START_POINTER_B: u32;
override START_POINTER_O: u32;

// Module
@group(0) @binding(0)
var <storage, read_write> heap: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin (global_invocation_id) global_id: vec3<u32>,
){
    if global_id.x < LEN{
        let id = global_id.x;
        heap[START_POINTER_O + id] = heap[START_POINTER_A + id] + heap[START_POINTER_B + id];
    }
}
