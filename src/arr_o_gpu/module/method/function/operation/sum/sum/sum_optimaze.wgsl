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
    var data_len = pointer.y - pointer.x;
    if reduction_result[0] != 0.{
        data_len = u32(reduction_result[0]);
    }

    var out_reduction_len = (data_len + 512 - 1) / 512;

    // first summition
    var cache_len = 512u;
    if data_len < cache_len * (work_id.x + 1){
        cache_len = data_len - 512u * work_id.x;
    }

    let total_unit = (cache_len + 2 - 1) / 2;


    var sum = 0.;

    if local_id.x < total_unit{

        if reduction_result[0] == 0.{
            let start = global_id.x * 2;
            let end = start + 1;

            if (end - 512 * work_id.x) < cache_len{
                sum = heap[pointer.x + start] + heap[pointer.x + end];
            } else {
                sum = heap[pointer.x + start];
            }
        } else {
            let start = global_id.x * 2;
            let end = start + 1;

            // if reduction_result[0] != 0.{
            //     heap[61953 + local_id.x] = f32(end - 512 * work_id.x);
            // }

            if (end - 512 * work_id.x) < cache_len{
                sum = reduction_result[start + 1] + reduction_result[end + 1];
            } else {
                sum = reduction_result[start + 1];
            }

        }
    }

    workgroupBarrier();

    if local_id.x < total_unit{
        cache[local_id.x] = sum;
    }

    workgroupBarrier();

    // paralel reduction on cache
    if total_unit != 1{
        var cache_len = total_unit;
        var total_unit_reduction = (cache_len + 2 - 1) / 2;
        let total_reduction_iter = u32(ceil(log2(f32(cache_len))));

        for (var i = 0u; i < total_reduction_iter; i++){
            var sum = 0.;
            if local_id.x < total_unit_reduction{

                let start = local_id.x * 2;
                let end = start + 1;

                if end < cache_len{
                    sum = cache[start] + cache[end];
                } else {
                    sum = cache[start];
                }

            }

            workgroupBarrier();

            if local_id.x < total_unit_reduction{
                cache[local_id.x] = sum;
            }

            workgroupBarrier();

            cache_len =  total_unit_reduction;
            total_unit_reduction = (total_unit_reduction +  2 - 1) / 2;
        }
    }

    if local_id.x == 0 && out_reduction_len == 1{
        heap[pointer_o.x] = cache[0];
    } else if local_id.x == 0 {
        if work_id.x == 0{
            reduction_result[0] = f32(out_reduction_len);
        }
        reduction_result[work_id.x + 1] = cache[0];
        // heap[121 + 61953] = 99999.;
    }
}
