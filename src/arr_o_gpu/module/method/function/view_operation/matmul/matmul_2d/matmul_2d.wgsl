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

// workgroup / cache
// // tile a
var<workgroup> tile_a:array<array<f32, 16>, 16>;
// // tile b
var<workgroup> tile_b:array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) local_id:vec3<u32>,
    @builtin(workgroup_id) workgroup_id:vec3<u32>,
    @builtin(global_invocation_id) global_id:vec3<u32>
){
    let m = shape_a[0];
    let k = shape_a[1];
    let n = shape_b[1];

    let size = 16u;
    let row = local_id.x;
    let coll = local_id.y;
    let work_x = workgroup_id.x;
    let work_y = workgroup_id.y;

    let iteration = (k + size - 1) / size;
    var sum = 0.;
    for (var i = 0u; i < iteration; i++){
        // save to cache
        // // Array A
        let row_a = (work_x * size) + row;
        let coll_a = (i * size) + coll;
        if (row_a < m && coll_a < k){
            let index_a = indexing_array_a(row_a, coll_a) + pointer_a.x;
            tile_a[row][coll] = heap[index_a];
        } else {
            tile_a[row][coll] = 0.;
        }

        // // Array B
        let row_b = (i * size) + row;
        let coll_b = (work_y * size) + coll;
        if (row_b < k && coll_b < n){
            let index_b = indexing_array_b(row_b, coll_b) + pointer_b.x;
            tile_b[row][coll] = heap[index_b];
        } else {
            tile_b[row][coll] = 0.;
        }

        // sync
        workgroupBarrier();

        // compute tilled
        for (var ii = 0u; ii < size; ii++){
            let global_k = (i * size) + ii;
            let global_m = (work_x * size) + row;
            let global_n = (work_y * size) + coll;

            // overflow in k, m and n
            if (global_k >= k || global_m >= m || global_n >= n){break;}

            // operation
            let a = tile_a[row][ii];
            let b = tile_b[ii][coll];
            sum += a * b;
        }

        // sync
        workgroupBarrier();
    }

    // output
    if (global_id.x < m && global_id.y < n){
        let index = pointer_o.x + global_id.x * stride_o[0] + global_id.y * stride_o[1];
        heap[index] = sum;
    }
}

fn indexing_array_a(row:u32, coll:u32)-> u32{
    var index = offset_a + stride_a[0] * row + stride_a[1] * coll;
    return index;
}

fn indexing_array_b(row:u32, coll:u32)-> u32{
    var index = offset_b + stride_b[0] * row + stride_b[1] * coll;
    return index;
}