// override
override LEN: u32;
override START_POINTER: u32;
override START_POINTER_O: u32;

override SCALAR_INDEX: u32; // offset + pointer.x

override SCALAR: f32;
override SCALAR_COUNTER: f32; // 0 = number , 1 = array

@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
){
    if global_id.x < LEN{
        let scalar = SCALAR * (1. - SCALAR_COUNTER) + heap[SCALAR_INDEX] * SCALAR_COUNTER;
        heap[START_POINTER_O + global_id.x] = heap[START_POINTER + global_id.x] + scalar;
    }
}
