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

struct StaticInterface{
    counter: u32,
    scalar: f32,
    index: u32,
    padding: u32,
}

// MODULE
// // heap
@group(0) @binding(0)
var<storage, read_write> heap:array<f32>;

// // execute_args
@group(0) @binding(1)
var<uniform> execute_args: array<ArrayMetadata, 2>;
// 0 : array
// 1 : array output

// // // static_cache
@group(0) @binding(2)
var<uniform> static_cache: StaticInterface;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin (global_invocation_id) global_id: vec3<u32>
){
    if global_id.x < execute_args[0].len{
        var scalar = heap[static_cache.index];
        scalar = select( scalar, static_cache.scalar, static_cache.counter == 0);
        heap[execute_args[1].pointer.x + global_id.x] = heap[execute_args[0].pointer.x + indexing(global_id.x)] - scalar;

    }
}

fn indexing(x:u32) -> u32{
    let arr = execute_args[0];
    let shape = arr.shape;
    let dim = arr.dim;
    let stride = arr.stride;
    let o_stride = arr.o_stride;

    var idx = arr.offset;
    for (var i = 0u; i < dim; i++){
        let idx0 = i >> 2;
        let idx1 = i & 3;
        let permute = (x / o_stride[idx0][idx1]) % shape[idx0][idx1];
        idx += (permute * stride[idx0][idx1]);
    }
    return idx;
}
