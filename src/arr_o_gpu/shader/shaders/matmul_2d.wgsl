// heap
@group(0) @binding(0)
var <storage, read_write> heap: array<f32>;

// array A
@group(1) @binding(0)
var <storage, read> pointer_a: vec2<u32>;

@group(1) @binding(1)
var <storage, read> shape_a: vec2<u32>;

@group(1) @binding(2)
var <storage, read> stride_a: vec2<u32>;

// array B
@group(1) @binding(3)
var <storage, read> pointer_b: vec2<u32>;

@group(1) @binding(4)
var <storage, read> shape_b: vec2<u32>;

@group(1) @binding(5)
var <storage, read> stride_b: vec2<u32>;

// output
@group(1) @binding(6)
var <storage, read_write> output: array<f32>;

@group(1) @binding(7)
var <storage, read> stride_output: vec2<u32>;

// tiling
var <workgroup> tile_a: array<array<f32, 2>, 2>;
var <workgroup> tile_b: array<array<f32, 2>, 2>;

@compute @workgroup_size(2, 2, 1)
fn main(
    @builtin(local_invocation_id) local_id:vec3<u32>,
    @builtin(workgroup_id) workgroup_id:vec3<u32>
    ){

    let m = shape_a.x;
    let n = shape_b.y;
    let k = shape_b.x;

    let size = 2u;

    // looping = k / size
    let looping = u32(ceil(f32(k) / f32(size)));

    // group_id
    let group_x = workgroup_id.x;
    let group_y = workgroup_id.y;

    var acc = 0.0;
    var over = false;
    for (var i = 0u; i < looping; i++){
        // cache 
        let x_a = local_id.x + (size * group_x);
        let y_a = local_id.y + (size * i);
        if (shape_a.x <= x_a || shape_a.y <= y_a){
            over = true;
            break;
        }
        let index_a = pointer_a.x + pointing(x_a, y_a, stride_a);
        tile_a[local_id.x][local_id.y] = heap[index_a];

        let x_b = local_id.x + (size * i);
        let y_b = local_id.y + (size * group_y);
        if (shape_b.x <= x_b || shape_b.y <= y_b){
            over = true;
            break;
        }
        let index_b = pointer_b.x + pointing(x_b, y_b, stride_b);
        tile_b[local_id.x][local_id.y] = heap[index_b];

        workgroupBarrier();

        // matmul algo
        for(var t_k = 0u; t_k < size; t_k++){
            let x = group_x * size;
            let row = x + t_k;
            if (shape_b.x <= row){
                break;
            }

            let y = group_y * size;
            let coll = y + t_k;
            if (shape_a.y <= coll){
                break;
            }

            acc += tile_a[local_id.x][t_k] * tile_b[t_k][local_id.y];
        }

        workgroupBarrier();
    }

    if (!over){
        let x = local_id.x + (group_x * size);  
        let y = local_id.y + (group_y * size);
        let output_index = pointing(x, y, stride_output);
        output[output_index] = acc;
    }
}

fn pointing(x:u32, y:u32, stride:vec2<u32>) -> u32{
    return x * stride.x + y * stride.y;
}