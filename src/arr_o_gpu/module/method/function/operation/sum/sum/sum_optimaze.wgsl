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

// //
@group(3) @binding(0)
var<storage, read_write> reduction_result: array<f32>;

// workgroup
var<workgroup> cache: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id:vec3<u32>,
    @builtin(local_invocation_id) local_id:vec3<u32>,
    @builtin(workgroup_id) work_id: vec3<u32>,
){
    heap[512] = 99999.;
    var data_len = pointer.y - pointer.x;
    var out_reduction_len = (data_len + 512 - 1) / 512;

    // first summition
    var cache_len = 512u;
    if data_len < cache_len * (work_id.x + 1){
        cache_len = cache_len - 512u * work_id.x;
    }

    let total_unit = (data_len + 2 - 1) / 2;
    var sum = 0.;
    if local_id.x < total_unit{

        let start = global_id.x * 2;
        let end = start + 1;
        if end < cache_len{
            sum = heap[pointer.x + start] + heap[pointer.x + end];
        } else {
            sum = heap[pointer.x + start];
        }
    }

    workgroupBarrier();

    if local_id.x < total_unit{
        cache[local_id.x] = sum;
    }

    workgroupBarrier();

    // paralel reduction
    if total_unit != 1{
        var cache_len = total_unit;
        var total_unit = (cache_len + 2 - 1) / 2;
        let total_reduction_iter = u32(ceil(log2(f32(cache_len))));

        for (var i = 0u; i < total_reduction_iter; i++){
            var sum = 0.;
            if local_id.x < total_unit{
                let start = global_id.x * 2;
                let end = start + 1;
                if end < cache_len{
                    sum = cache[start] + cache[end];
                } else {
                    sum = cache[start];
                }

            }

            workgroupBarrier();

            if local_id.x < total_unit{
                cache[local_id.x] = sum;
            }

            workgroupBarrier();

            cache_len =  total_unit;
            total_unit = (total_unit +  2 - 1) / 2;
        }
    }

    if local_id.x == 0{
    heap[513] = cache[0];

    }
}