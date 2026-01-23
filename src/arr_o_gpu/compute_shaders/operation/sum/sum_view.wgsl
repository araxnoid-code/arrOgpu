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

// MODULE
// // heap
@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;
// // execute_args
@group(0) @binding(1)
var<uniform> execute_args: array<ArrayMetadata, 3>;

// reduction
struct Counter{
    value: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
    padding3: array<vec4<u32>, 15>
}

// // reduction_heap
@group(1) @binding(0)
var<storage, read_write> reduction_heap: array<f32>;

@group(1) @binding(1)
var<uniform> reduction_len_list: Counter;

@group(1) @binding(2)
var<uniform> reduction_counter: Counter;

// heap
var<workgroup> cache: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin (global_invocation_id) global_id: vec3<u32>,
    @builtin (local_invocation_id) local_id: vec3<u32>,
    @builtin (workgroup_id) work: vec3<u32>,
){
    let total_thread = 256u;
    let total_thread_double = 512u;
    var len = select(reduction_len_list.value, execute_args[0].len, reduction_counter.value == 0);
    var reduction_len = select(total_thread_double, len - total_thread_double * work.x, total_thread_double * (work.x + 1) > len);

    let used_thread = (reduction_len + 1) >> 1;
    if local_id.x < used_thread{
        let start = global_id.x * 2;
        let end = start + 1;

        if reduction_counter.value == 0{
            if end - work.x * 512 < reduction_len{
                cache[local_id.x] = heap[execute_args[0].pointer.x + pointing_a(start)] + heap[execute_args[0].pointer.x + pointing_a(end)];

            } else {
                cache[local_id.x] = heap[execute_args[0].pointer.x + pointing_a(start)];
            }
        } else {
            let counter = u32(work.x * 512 < reduction_len);
            let value = select(reduction_heap[start], reduction_heap[start] + reduction_heap[end * counter], counter == 1);

            if end - work.x * 512 < reduction_len{
                cache[local_id.x] = reduction_heap[start] + reduction_heap[end];
            } else {
                cache[local_id.x] = reduction_heap[start];
            }
        }
    } else {
        cache[local_id.x] = 0.;
    }
    workgroupBarrier();

    var stride = 1u;
    while stride < total_thread {
        let start = local_id.x * stride * 2;
        let end = start + stride;

        let condition = end < total_thread;
        let idx = select(0, end, condition);
        cache[start] += select(0., cache[idx], condition);

        stride <<= 1;
        workgroupBarrier();
    }

    if local_id.x != 0{
        return;
    }


    if (len + 511) >> 9 == 1{
        heap[execute_args[1].pointer.x] = cache[0];
    } else {
        reduction_heap[work.x] = cache[0];
    }
}

fn pointing_a(x: u32) -> u32{
    let arr = execute_args[0];
    var idx = arr.offset;
    for(var i = 0u; i < arr.dim; i++){
        var idx0 = i >> 2;
        var idx1 = i & 3;
        let permute = (x / arr.o_stride[idx0][idx1]) % arr.shape[idx0][idx1];
        idx += permute * arr.stride[idx0][idx1];
    }
    return idx;
}

fn pointing_b(x: u32) -> u32{
    let arr = execute_args[1];
    var idx = arr.offset;
    for(var i = 0u; i < arr.dim; i++){
        var idx0 = i >> 2;
        var idx1 = i & 3;
        let permute = (x / arr.o_stride[idx0][idx1]) % arr.shape[idx0][idx1];
        idx += permute * arr.stride[idx0][idx1];
    }
    return idx;
}
