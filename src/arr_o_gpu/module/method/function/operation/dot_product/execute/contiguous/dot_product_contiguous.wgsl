// override
override LEN: f32;
override START_POINTER_A: u32;
override START_POINTER_B: u32;
override START_POINTER_OUT: u32;

// module
// // heap
@group(0) @binding(0)
var<storage, read_write> heap:array<f32>;

// reduction
struct Counter{
    value: u32,
    padding: array<u32, 63>,
}

// // reduction_cache
@group(1) @binding(0)
var<storage, read_write> reduction: array<f32>;

// // counter
@group(1) @binding(1)
var<storage, read> reduction_counter: Counter;

// // len
@group(1) @binding(2)
var<storage, read> len_reduction: Counter;

// cache
var<workgroup> cache: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) work_id: vec3<u32>,
){
    let total_thread = 256u;
    let total_thread_double = total_thread * 2u;
    let data_len: u32 = u32(mix(LEN, f32(len_reduction.value), f32(reduction_counter.value)));

    var cache_len: u32;
    if data_len < total_thread_double * (work_id.x + 1) {
        cache_len = data_len - total_thread_double * work_id.x;
    } else {
        cache_len = total_thread_double;
    }


    let total_unit = (cache_len + 1) / 2;
    var sum = 0.;
    if local_id.x < total_unit{
        let index = local_id.x + total_thread * work_id.x;
        let start = index * 2;
        let end = start + 1;


        if reduction_counter.value == 0{
            if (local_id.x * 2) + 1 < cache_len{
                let res_a = heap[START_POINTER_A + start] * heap[START_POINTER_B + start];
                let res_b = heap[START_POINTER_A + end] * heap[START_POINTER_B + end];
                sum = res_a + res_b;
            } else {
                let res = heap[START_POINTER_A + start] * heap[START_POINTER_B + start];
                sum = res;
            }
        } else {
            if (local_id.x * 2) + 1 < cache_len{
                sum = reduction[start] + reduction[end];
            } else {
                sum = reduction[start];
            }
        }

        cache[local_id.x] = sum;
    }
    workgroupBarrier();

    var stride = 1u;
    while (stride < total_thread){
        let index = local_id.x * 2 * stride;
        if index + stride < total_thread {
            cache[index] += cache[index + stride];
        }
        stride <<= 1u;
        workgroupBarrier();
    }

    let out_len = (data_len + total_thread_double - 1) / total_thread_double;
    if out_len == 1 && local_id.x == 0{
        heap[START_POINTER_OUT] = cache[0];
    } else if local_id.x == 0 {
        reduction[work_id.x] = cache[0];
    }
}
