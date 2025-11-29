// heap
@group(0) @binding(0)
var <storage, read_write> heap: array<f32>;

// A
// // pointer
@group(1) @binding(0)
var <uniform> pointer_a: vec2<u32>;

// // stride of matrix
@group(1) @binding(1)
var <uniform> matrix_stride_a: vec2<u32>;

// // shape of matrix
@group(1) @binding(2)
var <uniform> matrix_shape_a: vec2<u32>;

// B
// // pointer
@group(1) @binding(3)
var <uniform> pointer_b: vec2<u32>;

// // stride of matrix
@group(1) @binding(4)
var <uniform> matrix_stride_b: vec2<u32>;

// // shape of matrix
@group(1) @binding(5)
var <uniform> matrix_shape_b: vec2<u32>;

// output
// // pointer
@group(1) @binding(6)
var <uniform> pointer_out: vec2<u32>;

// // stride of matrix
@group(1) @binding(7)
var <uniform> matrix_stride_out: vec2<u32>;

// other
@group(1) @binding(8)
var <uniform> stride_between_matrix_a: u32; // = length of matrix a

@group(1) @binding(9)
var <uniform> stride_between_matrix_b: u32; // = length of matrix b

@group(1) @binding(10)
var <uniform> stride_between_matrix_out: u32; // = length of output matrix


// workgroup
// // tile
// // // a
var <workgroup> tile_a: array<array<f32, 16>, 16>;

// // // b
var <workgroup> tile_b: array<array<f32, 16>, 16>;

// description
// n-dimensional array multiplication function using tiling with a size of 16x16
@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id:vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id:vec3<u32>
){
    // init
    // i, j, m, k * i, j, k, n
    let m = matrix_shape_a.x;
    let k = matrix_shape_a.y;
    let n = matrix_shape_b.y;
    let size = 16u;
    let total_group_loop = (k + size - 1) / size;

    // range of matrix on Array
    // useful in jumping to the target matrix
    let start_a = global_id.z * stride_between_matrix_a;
    let start_b = global_id.z * stride_between_matrix_b;
    
    var acc = 0.;
    for (var i = 0u; i < total_group_loop; i++){
        // cache A
        let index_x_a = local_id.x + (group_id.x * size);
        let index_y_a = local_id.y + (i * size);

        // get the value of array a that corresponds to the tiling range and store it in the cache tile_a
        if (index_x_a < matrix_shape_a.x && index_y_a < matrix_shape_a.y){
            let index_a = pointer_a.x + indexing_pointer(start_a, index_x_a, index_y_a, matrix_stride_a);
            tile_a[local_id.x][local_id.y] = heap[index_a];
        } else {
            tile_a[local_id.x][local_id.y] = 0.;
        }

        // cache B
        let index_x_b = local_id.x + (i * size);
        let index_y_b = local_id.y + (group_id.y * size);

        // get the value of array a that corresponds to the tiling range and store it in the cache tile_b
        if (index_x_b < matrix_shape_b.x && index_y_b < matrix_shape_b.y){
            let index_b = pointer_b.x + indexing_pointer(start_b, index_x_b, index_y_b, matrix_stride_b);
            tile_b[local_id.x][local_id.y] = heap[index_b];
        } else {
            tile_b[local_id.x][local_id.y] = 0.;
        }

        // sync thread
        workgroupBarrier();

        // the limits of arrays a and b
        let array_a_k = group_id.x * size + local_id.x;
        let array_b_k = group_id.y * size + local_id.y;
        // matrix operation between tile_a and tile_b
        for (var ii = 0u; ii < size; ii++){
            var _k = ii + i * size;

            // indexing over 
            if (_k >= k || array_a_k >= m || array_b_k >= n){break;}
            
            acc += tile_a[local_id.x][ii] * tile_b[ii][local_id.y];
        }

        // sync thread
        workgroupBarrier();
    }

    let x = local_id.x + size * group_id.x;
    let y = local_id.y + size * group_id.y;
    if (x < m && y < n){
        let start_out = global_id.z * stride_between_matrix_out;
        let indexing = pointer_out.x + indexing_pointer(start_out, x, y, matrix_stride_out);
        heap[indexing] = acc;
    }
}

// description
// determine linear index based on index array n-dimension
fn indexing_pointer(start:u32, x:u32, y:u32, stride:vec2<u32>)-> u32{
    return start + x * stride.x + y * stride.y;
}