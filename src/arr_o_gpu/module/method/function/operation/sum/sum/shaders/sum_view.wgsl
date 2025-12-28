// heap
@group(0) @binding(0)
var<storage, read_write> heap:array<f32>;

// array A
// // pointer
@group(1) @binding(0)
var<uniform> pointer: vec2<u32>;

// // shape
@group(1) @binding(1)
var<storage, read> shape: array<u32>;

// // iters
@group(1) @binding(2)
var<storage, read> iters: array<u32>;

// // stride
@group(1) @binding(3)
var<storage, read> stride: array<u32>;

// // offset
@group(1) @binding(4)
var<uniform> offset: u32;

// output
// // pointer
@group(2) @binding(0)
var<uniform> pointer_o: vec2<u32>;

// // shape
@group(2) @binding(1)
var<storage, read> shape_o: array<u32>;

// // iters
@group(2) @binding(2)
var<storage, read> iters_o: array<u32>;

// // stride
@group(2) @binding(3)
var<storage, read> stride_o: array<u32>;

// // offset
@group(2) @binding(4)
var<uniform> offset_o: u32;

// // others
struct ReducCounter{
    counter: u32,
    padding: array<u32, 63>
}

@group(3) @binding(0)
var<storage, read_write> reduction_result: array<f32>;

@group(3) @binding(1)
var<storage, read> reduction_counter: ReducCounter;

@group(3) @binding(2)
var<storage, read> reduction_len: ReducCounter;

// workgroup
var<workgroup> cache: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id:vec3<u32>,
    @builtin(workgroup_id) work_id: vec3<u32>,
){
   var data_len: u32;
    if reduction_counter.counter == 0 {
        data_len = get_len();
    } else {
        data_len = reduction_len.counter;
    }

    var cache_len: u32;
    if data_len < 512 * (work_id.x + 1){
        cache_len = data_len - 512 * work_id.x;
    } else {
        cache_len = 512;
    }

    var sum:f32;
    if local_id.x < cache_len{
        let index = local_id.x + 256 * work_id.x;
        let start = index * 2;
        let end = start + 1;


        if end < cache_len + 512 * work_id.x{
            if reduction_counter.counter == 0{
                sum = heap[pointer.x + indexing(start)] + heap[pointer.x + indexing(end)];
            } else {
                sum = reduction_result[start] + reduction_result[end];
            }
        } else {
            if reduction_counter.counter == 0{
                sum = heap[pointer.x + indexing(start)];
            } else {
                sum = reduction_result[start];
            }
        }
        cache[local_id.x] = sum;
    }
    workgroupBarrier();

    // paralel reduction
    cache_len = (cache_len + 2 - 1) / 2;
    let total_reduction_iter = u32(ceil(log2(f32(cache_len))));

    for (var i:u32 = 0; i < total_reduction_iter; i++){
        var sum = 0.;
        if local_id.x < 128{
            let start = local_id.x * 2;
            let end = start + 1;

            sum = cache[start] + cache[end];
        }

        workgroupBarrier();

        if local_id.x < 128{
            cache[local_id.x] = sum;
        }

        workgroupBarrier();
    }

    let out_length = (data_len + 511) / 512;
    if out_length == 1 && local_id.x == 0{
        heap[pointer_o.x] = cache[0];

    } else if local_id.x == 0{
        reduction_result[work_id.x] = cache[0];
    }
}

fn get_len() -> u32{
    let len_shape = arrayLength(&shape);
    var len = 1u;
    for (var i = 0u; i < len_shape; i++){
        len *= shape[i];
    }
    return len;
}

fn indexing(x: u32) -> u32{
    var idx = offset;
    let len = arrayLength(&shape);
    for (var i = 0u; i < len; i++){
        let permute = (x / iters[i]) % shape[i];
        idx += (permute * stride[i]);
    }
    return idx;
}
