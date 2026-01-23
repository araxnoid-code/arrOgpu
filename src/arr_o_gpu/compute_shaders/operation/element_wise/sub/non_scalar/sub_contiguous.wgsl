// INIT
struct ArrayMetadata32{
	pointer: vec2<u32>, // 8
	len: u32, // 4
	offset: u32, // 4
	dim: u32, // 4
	padding0: u32, // 4
	padding1: u32, // 4
	padding2: u32, // 4
}

// Module
@group(0) @binding(0)
var <storage, read_write> heap: array<f32>;

@group(0) @binding(1)
var <uniform> execute_args: array<ArrayMetadata32, 3>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin (global_invocation_id) global_id: vec3<u32>,
){
    if global_id.x < execute_args[0].len{
        let id = global_id.x;
        heap[execute_args[2].pointer.x + id] = heap[execute_args[0].pointer.x + id] - heap[execute_args[1].pointer.x + id];
    }
}
