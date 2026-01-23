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
	m_n_o_stride: array<vec4<u32>, 2>, // 32
	padding3: array<vec4<u32>, 4>,
}

// MODULE
// // heap
@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;
// // execute_args
@group(0) @binding(1)
var<uniform> execute_args: array<ArrayMetadata, 3>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin (global_invocation_id) global_id: vec3<u32>
){
    if global_id.x >= execute_args[0].len{
        return;
    }

    let dim = execute_args[0].dim;
    var idx = execute_args[0].offset;
    for (var i = 0u; i < dim; i++){
        var idx0 = i >> 2;
        var idx1 = i & 3;
        let permute = (global_id.x / execute_args[0].o_stride[idx0][idx1]) % execute_args[0].shape[idx0][idx1];
        idx += permute * execute_args[0].stride[idx0][idx1];
    }

    heap[global_id.x + execute_args[1].pointer.x] = heap[idx + execute_args[0].pointer.x];
}
