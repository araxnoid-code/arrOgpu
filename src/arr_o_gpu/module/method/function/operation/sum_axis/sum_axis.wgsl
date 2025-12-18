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

// others
// // list of axis
@group(3) @binding(0)
var<storage, read> axis_list: array<u32>;

// // shape_of_slice
@group(3) @binding(1)
var<storage, read> shape_of_slice: array<u32>;

// // stride_of_slice
@group(3) @binding(2)
var<storage, read> stride_of_slice: array<u32>;

// workgroup
var<workgroup> cache: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
){
    let global_x = global_id.x;

    let local_x = local_id.x;
    let local_y = local_id.y;

    let length_output = pointer_o.y - pointer_o.x;
    let axis_length = arrayLength(&axis_list);
    let shape_length = arrayLength(&shape);

    let total_value = get_total_value(axis_length);
    let size = 16u;
    let iteration_value = (total_value + size - 1) / size;

    var acc = 0.;
    for (var i = 0u; i < iteration_value; i++){
        // save value to cache
        let index = size * i + local_y;
        if index < total_value && global_x < length_output {
            var _offset = offset;
            var indexing = 0u;
            
            var idx = 0u;
            for(var ii = 0u; ii < shape_length; ii++){
                var in_axis = false;
                for(var iii = 0u; iii < axis_length; iii++){
                    if (ii == axis_list[iii]){
                        in_axis = true;
                        break;
                    }
                }
                if (in_axis){
                    let start_permute = (index / stride_of_slice[ii]) % shape_of_slice[ii];
                    indexing += start_permute * stride[ii]; 
                } else {
                    let start_permute = (global_x / iters_o[idx]) % shape_o[idx];
                    idx += 1;
                    _offset += start_permute * stride[ii];
                }
            }

            indexing += (_offset + pointer.x);
            cache[local_x][local_y] = heap[indexing];
        }

        workgroupBarrier();

        // paralel reduction
        var length_cache = size;
        if (i + 1) >= iteration_value {
            length_cache = total_value - size * i;
        }

        var total_unit = (length_cache + 2 - 1) / 2;


        while total_unit != 0 {
            var sum = 0.;
            if local_y < total_unit && global_x < length_output {
                let start = local_y * 2;
                
                let end = start + 1;
                if end < length_cache{
                    sum = cache[local_x][start] + cache[local_x][end];
                } else {
                    sum = cache[local_x][start];
                }
            }

            workgroupBarrier();

            if local_y < total_unit && global_x < length_output{
                cache[local_x][local_y] = sum;
            }

            workgroupBarrier();

            if total_unit > 1 {
                length_cache = total_unit;
                total_unit = (total_unit + 2 - 1) / 2;
            } else {
                total_unit = 0;
            }
        }

        if local_y == 0 && global_x < length_output{
            acc += cache[local_x][0];
        }
    }

    let index = pointer_o.x + global_x;
    if global_x < length_output && local_y == 0{
        heap[index] = acc;
    }
}

fn get_total_value(length:u32) -> u32{
    var product = 1u;
    for(var i = 0u; i < length; i++){
        product *= shape[axis_list[i]];
    }
    return product;   
}