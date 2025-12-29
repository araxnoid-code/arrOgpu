// init
// // metadata
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

// module
// // heap
@group(0) @binding(0)
var<storage, read_write> heap:array<f32>;

// // cache
// // // execute_array_cache
@group(0) @binding(1)
var<storage, read> execute_array_cache: array<ArrayMetaData>;


// array A
// // pointer
@group(1) @binding(0)
var<uniform> pointer_a: vec2<u32>;

// array B
// // pointer
@group(2) @binding(0)
var<uniform> pointer_b: vec2<u32>;

// output
// // pointer
@group(3) @binding(0)
var<uniform> pointer_o: vec2<u32>;

// reduction
struct Counter{
    value: u32,
    padding: array<u32, 63>,
}

// // reduction_cache
@group(4) @binding(0)
var<storage, read_write> reduction: array<f32>;

// // counter
@group(4) @binding(1)
var<storage, read> reduction_counter: Counter;

// // len
@group(4) @binding(2)
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
    let data_len: u32 = u32(mix(f32(pointer_a.y - pointer_a.x), f32(len_reduction.value), f32(reduction_counter.value)));

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
                let res_a = heap[pointer_a.x + start] * heap[pointer_b.x + start];
                let res_b = heap[pointer_a.x + end] * heap[pointer_b.x + end];
                sum = res_a + res_b;
            } else {
                let res = heap[pointer_a.x + start] * heap[pointer_b.x + start];
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
        heap[pointer_o.x] = cache[0];
    } else if local_id.x == 0 {
        reduction[work_id.x] = cache[0];
    }
}
