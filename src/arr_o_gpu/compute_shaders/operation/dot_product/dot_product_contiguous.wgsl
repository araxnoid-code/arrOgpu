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

// // reduction_heap
@group(1) @binding(0)
var<storage, read_write> reduction_heap: array<f32>;

@group(1) @binding(0)
var<uniform> reduction_counter: u32;

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
    let len = execute_args[0].len;
    let output_len = (len + 511) >> 9;

    var reduction_len = total_thread_double;
    if total_thread_double * (work.x + 1) > len{
        reduction_len = len - total_thread_double * work.x;
    }



    if global_id.x < len && global_id.x < reduction_len{
        let start = global_id.x * 2;
        let end = start + 1;
        if end < reduction_len{
            let a = heap[execute_args[0].pointer.x + start] * heap[execute_args[1].pointer.x + start];
            let b = heap[execute_args[0].pointer.x + end] * heap[execute_args[1].pointer.x + end];
            cache[local_id.x] = a + b;

        } else {
            let a = heap[execute_args[0].pointer.x + start] * heap[execute_args[1].pointer.x + start];
            cache[local_id.x] = a;
        }
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

    if output_len == 1 && local_id.x == 0{
        heap[execute_args[2].pointer.x] = cache[0];
    } else if local_id.x == 0 {
        reduction_heap[work.x] = cache[0];
    }
}
