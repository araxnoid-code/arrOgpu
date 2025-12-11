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
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) work_id: vec3<u32>,
    @builtin(global_invocation_id) global_id:vec3<u32>,
){
    let len_of_shape = arrayLength(&shape_a);
    let z = global_id.z;

    if (z < total_of_matrix(len_of_shape)){
        let m = shape_a[len_of_shape - 2]
        let k = shape_b[len_of_shape - 2];
        let n = shape_b[len_of_shape - 1];
    
        let row = local_id.x;
        let coll = local_id.y;
        let work_x = work_id.x;
        let work_y = work_id.y;
        let size = 16u;


        let iteration = (k + size - 1) / size;
        var sum = 0.;
        for (var i = 0u; i < iteration; i++){
            // get cache
            // // array a
            let row_a = (size * work_x) + row;
            let coll_a = (size * i) + coll;
            if (row_a < m && coll_a < k){
                let index = indexing_array_a(row_a, coll_a, len_of_shape, z) + pointer_a.x;
                tile_a[row][coll] = heap[index];
            } else {
                tile_a[row][coll] = 0.;
            }

            // // array b
            let row_b = (size * i) + row;
            let coll_b = (size * work_y) + coll;
            if(row_b < k && coll_b < n){
                let index = indexing_array_b(row_b, coll_b, len_of_shape, z) + pointer_b.x;
                tile_b[row][coll] = heap[index];
            } else {
                tile_b[row][coll] = 0.;
            }

            // sync
            workgroupBarrier();

            // compute
            let global_m = (size * work_x) + row;
            let global_n = (size * work_y) + coll;
            for (var ii = 0u; ii < size, ii++){
                let global_k = (size * i) + ii;

                if (global_k >= k || global_m >= m || global_n >= n){break;}
                sum += tile_a[row][ii] * tile_b[ii][coll]; 
            }

            // sync
            workgroupBarrier();
        }

        if (global_id.x < m && global_id.y < n){
            let index = shape_o[len_of_shape - 2] * global_id.x + shape_o[len_of_shape - 1] * global_id.y + stride_o[len_of_shape - 3] * global_id.z;
            heap[index] = sum;
        }
    }
}

fn indexing_array_a(row:u32, coll:u32, len:u32, z:u32) -> u32{
    let index = offset_a + shape_a[len - 2] * row + shape_a[len - 1] * coll + stride_a[len - 3] * z;
    return index;
}

fn indexing_array_b(row:u32, coll:u32, len:u32, z:u32) -> u32{
    let index = offset_b + shape_b[len - 2] * row + shape_b[len - 1] * coll + stride_b[len - 3] * z;
    return index;
}

fn total_of_matrix(len:u32) -> u32{
    var product = 1;
    for(var i = 0u; i < len - 2; i++){
        product *= shape_o[i];
    }
    return product;
}
