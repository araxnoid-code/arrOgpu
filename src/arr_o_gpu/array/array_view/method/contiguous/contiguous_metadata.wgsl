// init
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

struct ExecuteArgs{
    arg0: ArrayMetadata,
    arg1: ArrayMetadata,
    arg2: ArrayMetadata,
}

// override
override POINTER_START: u32;
override POINTER_START_OUTPUT: u32;
override LEN: u32;
override DIM: u32;
override OFFSET: u32;

// module
// // heap
@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;

// // cache
@group(0) @binding(1)
var<uniform> execute_args: ExecuteArgs;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    if global_id.x < LEN{
        let index = POINTER_START + indexing(global_id.x);
        heap[POINTER_START_OUTPUT + global_id.x] = heap[index];
    }
}

fn indexing(x: u32) -> u32{
    var index = OFFSET;
    for (var i = 0u; i < DIM; i++){
        let permute = x / array_access(i, execute_args.arg0.o_stride) % array_access(i, execute_args.arg0.shape);
        index += (permute * array_access(i, execute_args.arg0.stride));
    }
    return index;
}

fn array_access(index:u32, arr: array<vec4<u32>, 2>) -> u32{
    return arr[index >> 2][index & 3];
}
