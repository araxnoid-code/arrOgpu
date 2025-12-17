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

// others
// // list of axis
@group(4) @binding(0)
var<storage, read> axis_list: array<f32>;

// // shape_of_slice
@group(4) @binding(1)
var<storage, read> shape_of_slice: array<f32>;

// // stride_of_slice
@group(4) @binding(2)
var<storage, read> stride_of_slice: array<f32>;

// workgroup
var<workgroup> cache: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
){
    let global_x = global_id.x;
    let global_y = global_id.y;

    let local_x = global_id.x;
    let local_y = global_id.y;

    let length_output = pointer_o.y - pointer_o.x;
    if (global_x < length_output){
        let axis_length = arrayLength(&axis_list);
        let shape_length = arrayLength(shape);

        let total_value = get_total_value();
        let size = 16;
        let iteration_value = (total_value + size - 1) / size;

        for (var i = 0; i < iteration_value; i++){
            // save value to cache
            let index = size * i + local_y;
            if index < total_value {
                var offset = offset;
                var indexing = 0;

                var idx = 0;
                for(var i = 0; i < shape_length; i++){
                    var in_axis = false;
                    for(var ii = 0; ii < axis_length; ii++){
                        if (shape[i] == axis_list[ii]){
                            in_axis = true;
                            break;
                        }
                    }
                    if (in_axis){
                        let start_permute = (index / stride_of_slice[i]) % shape_of_slice[i];
                        indexing += start_permute * stride[i]; 
                    } else {
                        let start_permute = (global_x / iters_o[idx]) % shape_o[idx];
                        idx += 1;
                        offset += start_permute * stride[i];
                    }
                }
                indexing += offset;

                cache[local_x][local_y] = heap[indexing];
            }

            // sync
            workgroupBarrier();

            // paralel reduction
            var length_cache = size;
            if (i + 1) < iteration_value {
                length_cache = total_value - size * i;
            }
            let total_unit = (length_cache + 2 - 1) / 2;

        }
    }
}

fn get_total_value(length:u32) -> u32{
    var product = 1;
    for(var i = 0; i < length; i++){
        product *= shape[axis_list[i]];
    }
    return product;   
}