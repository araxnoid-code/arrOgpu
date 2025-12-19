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

// workgroup
var<workgroup> cache: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) local_id:vec3<u32>){
    let len_output = pointer_o.y - pointer_o.x;
    let len_array = pointer.y - pointer.x;

    var acc = 0.;
    let iteration = (len_array + 256 - 1) / 256;
    for (var i = 0u; i < iteration; i++){
        // stored data to cache
        let index = i * 256u + local_id.x;
        if index < len_array{
            cache[local_id.x] = heap[pointer.x + pointing(local_id.x)];
        }
        workgroupBarrier();

        // paralel reduction
        var len_cache = 256u;
        if i + 1 == iteration{
            len_cache = len_array - 256u * i;
        }
        var total_unit = (len_cache + 2 - 1) / 2;
        while total_unit != 0 {
            var sum = 0.;
            if local_id.x < total_unit{
                let start = local_id.x * 2;
                let end = start + 1;

                if end < len_cache{
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
                len_cache = total_unit;
                total_unit = (len_cache + 2 - 1) / 2;
            } else {
                total_unit = 0;
            }
        }

        if local_id.x == 0{
            acc += cache[0];
        }
    }

    if local_id.x == 0{
        heap[pointer_o.x] = acc;
    }
}

fn pointing(x:u32) -> u32{
    var index = offset;
    let len = arrayLength(&shape);
    for (var i = 0u; i < len; i++){
        let permute = (x / iters[i]) % shape[i];
        index += (permute * stride[i]);
    }
    return index;
}