// init
struct ArrayMetaData{
	pointer: vec2<u32>,
	len: u32,
	offset: u32,
	dim: u32,
	shape: array<u32, 10>,
	stride: array<u32, 10>,
	origin_stride: array<u32, 10>,
	padding: array<u32, 29>,
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
var<storage, read> execute_cache: array<ArrayMetaData, 3>;

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
        let permute = (x / execute_cache[0].origin_stride[i]) % execute_cache[0].shape[i];
        index += (permute * execute_cache[0].stride[i]);
    }

    return index;
}
