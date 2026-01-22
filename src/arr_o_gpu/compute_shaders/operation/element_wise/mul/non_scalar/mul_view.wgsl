// INIT
struct ArrayMetadata{
	pointer: vec2<u32>, // 8
	len: u32, // 4
	offset: u32, // 4
	dim: u32, // 4
	padding0: u32, // 4
	padding1: u32, // 4
	padding2: u32, // 4
	shape: array<vec4<u32>, 2>, // 32
	stride: array<vec4<u32>, 2>, // 32
	o_stride: array<vec4<u32>, 2>, // 32
	m_n_shape: array<vec4<u32>, 2>, // 32
	m_n_origin_stride: array<vec4<u32>, 2>, // 32
	padding3: array<vec4<u32>, 4>,
}

struct ExecuteArgs{
    arg0: ArrayMetadata,
    arg1: ArrayMetadata,
    arg2: ArrayMetadata,
}

// Module
// // heap
@group(0) @binding(0)
var <storage, read_write> heap: array<f32>;
// // execute args
@group(0) @binding(1)
var <uniform> execute_args: ExecuteArgs;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin (global_invocation_id) global_id: vec3<u32>,
){
    if global_id.x < execute_args.arg0.len{
    heap[execute_args.arg2.pointer.x + global_id.x] = heap[execute_args.arg0.pointer.x + indexing_a(global_id.x)] * heap[execute_args.arg1.pointer.x + indexing_b(global_id.x)];
    }
}

fn indexing_a(x: u32) -> u32{
    var idx = execute_args.arg0.offset;
    for (var i = 0u; i < execute_args.arg0.dim; i++){
        let permute = (x / execute_args.arg0.o_stride[i >> 2][i & 3]) % execute_args.arg0.shape[i >> 2][i & 3];
        idx += permute * execute_args.arg0.stride[i >> 2][i & 3];
    }

    return idx;
}

fn indexing_b(x: u32) -> u32{
    var idx = execute_args.arg1.offset;
    for (var i = 0u; i < execute_args.arg1.dim; i++){
        let permute = (x / execute_args.arg1.o_stride[i >> 2][i & 3]) % execute_args.arg1.shape[i >> 2][i & 3];
        idx += permute * execute_args.arg1.stride[i >> 2][i & 3];
    }
    return idx;
}
