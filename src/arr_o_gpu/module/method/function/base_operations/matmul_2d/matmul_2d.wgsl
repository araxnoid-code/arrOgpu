// heap
@group(0) @binding(0)
var <storage, read_write> heap: array<f32>;

// array A
// // pointer
@group(1) @binding(0)
var <uniform> pointer_a: vec2<u32>;

// // shape
@group(1) @binding(1)
var <uniform> shape_a: vec2<u32>;

// // stride
@group(1) @binding(2)
var <uniform> stride_a: vec2<u32>;

// array B
// // pointer
@group(1) @binding(3)
var <uniform> pointer_b: vec2<u32>;

// // shape
@group(1) @binding(4)
var <uniform> shape_b: vec2<u32>;

// // stride
@group(1) @binding(5)
var <uniform> stride_b: vec2<u32>;

// output
// // pointer
@group(2) @binding(0)
var <uniform> pointer_output: vec2<u32>;

// // stride
@group(2) @binding(1)
var <uniform> stride_output: vec2<u32>;

// workgroup
// // tiling
// // // a
var <workgroup> tile_a: array<array<f32, 16>, 16>;
// // // b
var <workgroup> tile_b: array<array<f32, 16>, 16>;

// description
// 2-dimensional array (matrix) multiplication function using tiling with a size of 16x16
@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) local_id:vec3<u32>,
    @builtin(workgroup_id) workgroup_id:vec3<u32>
){
    // init
    let size = 16u;
    let m = shape_a.x;
    let n = shape_b.y;
    let k = shape_b.x;
    let group_x = workgroup_id.x;
    let group_y = workgroup_id.y;

    let looping = (k + size - 1) / size;    // determine tiling movement
    var acc = 0.0;                          // will accumulate the results on each related tiling
    for (var i = 0u; i < looping; i++) {
        // a
        // tiling range on array a
        let x_a = local_id.x + (size * group_x);
        let y_a = local_id.y + (size * i);

        // b
        // tiling range on array b
        let x_b = local_id.x + (size * i);
        let y_b = local_id.y + (size * group_y);

        // get the value of array a that corresponds to the tiling range and store it in the cache tile_a
        if (x_a < shape_a.x && y_a < shape_a.y) {
            let index_a = pointer_a.x + pointing(x_a, y_a, stride_a);
            tile_a[local_id.x][local_id.y] = heap[index_a];
        } else {
            tile_a[local_id.x][local_id.y] = 0.0;
        }

        // get the value of array a that corresponds to the tiling range and store it in the cache tile_b
        if (x_b < shape_b.x && y_b < shape_b.y) {
            let index_b = pointer_b.x + pointing(x_b, y_b, stride_b);
            tile_b[local_id.x][local_id.y] = heap[index_b];
        } else {
            tile_b[local_id.x][local_id.y] = 0.0;
        }

        // sync thread
        workgroupBarrier();

        // matrix operation between tile_a and tile_b
        for (var t_k = 0u; t_k < size; t_k++) {
            let global_k = i * size + t_k;

            // over 
            if (global_k >= k) { break; }

            // accumulate
            acc += tile_a[local_id.x][t_k] * tile_b[t_k][local_id.y];
        }

        // sync thread
        workgroupBarrier();
    }

    // save the accumulated results as output
    let x = local_id.x + (group_x * size);
    let y = local_id.y + (group_y * size);
    if (x < n && y < m) {
        let output_index = pointing(x, y, stride_output);
        let heap_index = pointer_output.x + output_index;
        heap[heap_index] = acc;
    }
}

// description
// determine linear index based on index matrix
fn pointing(x:u32, y:u32, stride:vec2<u32>) -> u32{
    return x * stride.x + y * stride.y;
}