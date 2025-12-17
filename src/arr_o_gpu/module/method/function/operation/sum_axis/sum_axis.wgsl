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

// // out_shape_keep_dim
@group(4) @binding(1)
var<storage, read> shape_o_keep_dim: array<f32>;

// // out_shape_keep_dim
@group(4) @binding(2)
var<storage, read> stride_shape_o_keep_dim: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
){
    let x = global_id.x;
    let y = global_id.y;
    let length_output = pointer_o.y - pointer_o.x;
    if (x < length_output){
    // get slice representation
        let axis_length = arrayLength(&axis_list);
        let total_jump = get_total_jump();

        if (y < total_jump){
            let shape_length = arrayLength();
            var offset = offset;

            for(var i = 0; i < shape_length; i++){
                var in_axis = false;
                for(var ii = 0; ii < axis_length; ii++){
                    if (shape[i] == axis_list[ii]){
                        in_axis = true;
                        break;
                    }
                }

                if (in_axis){

                } else {
                    let start_permute =
                }

            }
        }

    }
}

fn get_total_jump(length:u32) -> u32{
    var product = 1;
    for(var i = 0; i < length; i++){
        product *= shape[axis_list[i]];
    }
    return product;   
}