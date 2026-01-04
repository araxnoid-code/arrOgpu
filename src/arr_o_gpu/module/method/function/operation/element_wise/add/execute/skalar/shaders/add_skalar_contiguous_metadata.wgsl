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

struct StaticInterface{
    counter: u32,
    scalar: f32,
    index: u32,
    padding: u32,
}

// MODULE
// // heap
@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;
// // execute_args
@group(0) @binding(1)
var<uniform> execute_args: array<ArrayMetadata32, 3>;
// // static cache
@group(0) @binding(2)
var<uniform> static_cache: StaticInterface;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
){
    if global_id.x < execute_args[0].len{
        var scalar = heap[static_cache.index];
        scalar = select( scalar, static_cache.scalar, static_cache.counter == 0);
        heap[execute_args[1].pointer.x + global_id.x] = heap[execute_args[0].pointer.x + global_id.x] + scalar;
    }
}
