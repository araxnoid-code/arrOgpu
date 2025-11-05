// heap
@group(0) @binding(0)
var <storage, read_write> heap: array<f32>;

// A
// pointer
@group(1) @binding(0)
var <uniform> pointer_a: vec2<u32>;

// stride of matrix
@group(1) @binding(1)
var <uniform> matrix_stride_a: vec2<u32>;

// shape of matrix
@group(1) @binding(2)
var <uniform> matrix_shape_a: vec2<u32>;

// B
// pointer
@group(1) @binding(3)
var <uniform> pointer_b: vec2<u32>;

// stride of matrix
@group(1) @binding(4)
var <uniform> matrix_stride_b: vec2<u32>;

// shape of matrix
@group(1) @binding(5)
var <uniform> matrix_shape_b: vec2<u32>;

// output
// pointer ouput
@group(1) @binding(6)
var <uniform> pointer_out: vec2<u32>;

// stride of matrix
@group(1) @binding(7)
var <uniform> matrix_stride_out: vec2<u32>;

// other
@group(1) @binding(8)
var <uniform> stride_between_matrix_a: u32; // = length of matrix a

@group(1) @binding(9)
var <uniform> stride_between_matrix_b: u32; // = length of matrix b

@group(1) @binding(10)
var <uniform> stride_between_matrix_out: u32; // = length of output matrix


// tile
var <workgroup> tile_a: array<array<f32, 16>, 16>;
var <workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id:vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id:vec3<u32>
){
    let m = matrix_shape_a.x;
    let k = matrix_shape_a.y; // i, j, m, k * i, j, k, n
    let n = matrix_shape_b.y;
    let total_group_loop = u32(ceil(f32(k) / 16.0));

    // range of matrix on Array
    let start_a = global_id.z * stride_between_matrix_a;
    let start_b = global_id.z * stride_between_matrix_b;
    
    let size = 16u;
    var acc = 0.;
    for (var i = 0u; i < total_group_loop; i++){
        // cache A
        let index_x_a = local_id.x + (group_id.x * size);
        let index_y_a = local_id.y + (i * size);
        if (index_x_a < matrix_shape_a.x && index_y_a < matrix_shape_a.y){
            let index_a = pointer_a.x + indexing_pointer(start_a, index_x_a, index_y_a, matrix_stride_a);
            tile_a[local_id.x][local_id.y] = heap[index_a];
        } else {
            tile_a[local_id.x][local_id.y] = 0.;
        }

        // cache B
        let index_x_b = local_id.x + (i * size);
        let index_y_b = local_id.y + (group_id.y * size);
        if (index_x_b < matrix_shape_b.x && index_y_b < matrix_shape_b.y){
            let index_b = pointer_b.x + indexing_pointer(start_b, index_x_b, index_y_b, matrix_stride_b);
            tile_b[local_id.x][local_id.y] = heap[index_b];
        } else {
            tile_b[local_id.x][local_id.y] = 0.;
        }

        workgroupBarrier();

        for (var ii = 0u; ii < size; ii++){
            var _k = ii + i * size;
            if (_k >= k){break;}
            acc += tile_a[local_id.x][ii] * tile_b[ii][local_id.y];
        }

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

fn indexing_pointer(start:u32, x:u32, y:u32, stride:vec2<u32>)-> u32{
    return start + x * stride.x + y * stride.y;
}