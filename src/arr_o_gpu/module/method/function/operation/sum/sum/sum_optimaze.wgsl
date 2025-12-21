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
    @builtin(local_invocation_id) local_id:vec3<u32>
){
    var data_len = pointer.y - pointer.x;
    if reduction_result[0] != 0.{
        data_len = u32(reduction_result[0]);
    }

    let out_len = (data_len + 512 - 1) / 512;

    var len_cache = 512u;
    if len_cache < (global_id.x + 1) * 512{
        len_cache = data_len - 512 * global_id.x;
    }
    
    // starting compute data
    // adding first to 256 for fit with cache
    var sum = 0.;
    var total_unit = (len_cache + 2 - 1) / 2;
    if global_id.x < total_unit{
        if reduction_result[0] == 0.{
            let start = global_id.x * 2;
            let end = start + 1;
            if end < len_cache{
                sum = heap[start + pointer.x] + heap[end + pointer.x];
            } else {
                sum = heap[start + pointer.x];
            } 
        } else {
            let start = (global_id.x * 2) + 1;
            let end = start + 1;
            if end < len_cache{
                sum = reduction_result[start] + reduction_result[end];
            } else {
                sum = reduction_result[start];
            }
        }
    }

    workgroupBarrier();

    if global_id.x < total_unit{
        cache[local_id.x] = sum;
    }

    len_cache = total_unit;

    workgroupBarrier();

    // paralel reduction
    if len_cache != 1 {
        var total_unit = (len_cache + 2 - 1) / 2;
        let reduction_iter = u32(ceil(log2(f32(len_cache))));
        for (var i = 0u; i < reduction_iter ; i++){
            var sum = 0.;
            if local_id.x < total_unit{
                if reduction_result[0] == 0.{
                    let start = local_id.x * 2;
                    let end = start + 1;

                    if end < len_cache{
                        sum = cache[start] + cache[end];
                    } else {
                        sum = cache[start];
                    } 
                } else {
                    let start = local_id.x * 2;
                    let end = start + 1;
                    if end < len_cache{
                        sum = cache[start] + cache[end];
                    } else {
                        sum = cache[start];
                    }
                }
            }

            workgroupBarrier();

            if local_id.x < total_unit{
                cache[local_id.x] = sum;
            }

            workgroupBarrier();

            len_cache = total_unit;
            total_unit = (total_unit + 2 - 1) / 2;
        }
    }


    if global_id.x == 0 && out_len == 1{
        heap[pointer_o.x] = cache[0];
    } else if global_id.x == 0 {
        reduction_result[global_id.x + 1] = cache[0];
        reduction_result[0] = f32(out_len);
    }
}