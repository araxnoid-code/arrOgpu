// heap
@group(0) @binding(0)
var<storage, read_write> heap:array<f32>;

// array A
// // pointer
@group(1) @binding(0)
var<uniform> pointer_a: vec2<u32>;

// // shape
@group(1) @binding(1)
var<storage, read> shape_a: array<u32>;

// // iters
@group(1) @binding(2)
var<storage, read> iters_a: array<u32>;

// // stride
@group(1) @binding(3)
var<storage, read> stride_a: array<u32>;

// // offset
@group(1) @binding(4)
var<uniform> offset_a: u32;

// array B
// // pointer
@group(2) @binding(0)
var<uniform> pointer_b: vec2<u32>;

// // shape
@group(2) @binding(1)
var<storage, read> shape_b: array<u32>;

// // iters
@group(2) @binding(2)
var<storage, read> iters_b: array<u32>;

// // stride
@group(2) @binding(3)
var<storage, read> stride_b: array<u32>;

// // offset
@group(2) @binding(4)
var<uniform> offset_b: u32;

// output
// // pointer
@group(3) @binding(0)
var<uniform> pointer_o: vec2<u32>;

// // shape
@group(3) @binding(1)
var<storage, read> shape_o: array<u32>;

// // iters
@group(3) @binding(2)
var<storage, read> iters_o: array<u32>;

// // stride
@group(3) @binding(3)
var<storage, read> stride_o: array<u32>;

// // offset
@group(3) @binding(4)
var<uniform> offset_o: u32;

// cache
var<workgroup> cache: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id:vec3<u32>,
){
    let length = shape_a[0];
    let size = 256u;

    let iterator = (length + size - 1) / size;
    var acc = 0.;
    for( var i = 0u; i < iterator; i++){
        let index = size * i + local_id.x;

        // mul
        if (index < length){
            let index_a = pointer_a.x + indexing_a(index);
            let index_b = pointer_b.x + indexing_b(index);
            cache[local_id.x] = heap[index_a] * heap[index_b];
        }

        // sync
        workgroupBarrier();

        // paralel reduction
        var cache_size = init_cache_size(i, length, size);
        var total_unit = (cache_size + 2 - 1) / 2;
        while total_unit != 0{
            var sum = 0.;
            if (local_id.x < total_unit){
                let start = local_id.x * 2;
                let end = start + 1;

                if (end < cache_size){
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

            if total_unit != 1{
                cache_size = total_unit;
                total_unit = (total_unit + 2 - 1) / 2;
            } else {
                total_unit = 0;
            }
        }

        // accumulate
        if local_id.x == 0{
            acc += cache[0];
        }
    }

    if local_id.x == 0{
        let heap_index = pointer_o.x;
        heap[heap_index] = acc;
    }

}

fn indexing_a(x:u32) -> u32{
    let index = offset_a + x * stride_a[0];
    return index;
}

fn indexing_b(x:u32) -> u32{
    let index = offset_b + x * stride_b[0];
    return index;
}

fn init_cache_size(i:u32, len:u32, size:u32) -> u32{
    let end = size * (i + 1);
    if (len >= end){
        return size;
    } else {
        return (len - size * i);
    }
}
