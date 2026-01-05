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

// MODULE
// // heap
@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;
// // execute_args
@group(0) @binding(1)
var<uniform> execute_args: array<ArrayMetadata32, 3>;




@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id:vec3<u32>
){
    if global_id.x < execute_args[0].len{
        heap[global_id.x + execute_args[1].pointer.x] = abs(heap[global_id.x + execute_args[0].pointer.x]);
    }
}
