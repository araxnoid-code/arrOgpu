// heap
@group(0) @binding(0)
var <storage, read_write> heap: array<f32>;

// array A
@group(1) @binding(0)
var <uniform> pointer_a: vec2<u32>;

@group(1) @binding(1)
var <uniform> shape_a: vec2<u32>;

@group(1) @binding(2)
var <uniform> stride_a: vec2<u32>;

// array B
@group(1) @binding(3)
var <uniform> pointer_b: vec2<u32>;

@group(1) @binding(4)
var <uniform> shape_b: vec2<u32>;

@group(1) @binding(5)
var <uniform> stride_b: vec2<u32>;

// output
// @group(2) @binding(0)
// var <storage, read_write> output: array<f32>;

@group(2) @binding(0)
var <uniform> pointer_output: vec2<u32>;

@group(2) @binding(1)
var <uniform> stride_output: vec2<u32>;

// tiling
var <workgroup> tile_a: array<array<f32, 16>, 16>;
var <workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) local_id:vec3<u32>,
    @builtin(workgroup_id) workgroup_id:vec3<u32>
){
    let m = shape_a.x;
    let n = shape_b.y;
    let k = shape_b.x;

    let size = 16u;
    let looping = u32(ceil(f32(k) / f32(size)));

    let group_x = workgroup_id.x;
    let group_y = workgroup_id.y;

    var acc = 0.0;

    for (var i = 0u; i < looping; i++) {
        let x_a = local_id.x + (size * group_x);
        let y_a = local_id.y + (size * i);
        let x_b = local_id.x + (size * i);
        let y_b = local_id.y + (size * group_y);

        if (x_a < shape_a.x && y_a < shape_a.y) {
            let index_a = pointer_a.x + pointing(x_a, y_a, stride_a);
            tile_a[local_id.x][local_id.y] = heap[index_a];
        } else {
            tile_a[local_id.x][local_id.y] = 0.0;
        }

        if (x_b < shape_b.x && y_b < shape_b.y) {
            let index_b = pointer_b.x + pointing(x_b, y_b, stride_b);
            tile_b[local_id.x][local_id.y] = heap[index_b];
        } else {
            tile_b[local_id.x][local_id.y] = 0.0;
        }

        workgroupBarrier();

        for (var t_k = 0u; t_k < size; t_k++) {
            let global_k = i * size + t_k;
            if (global_k >= k) { break; }
            acc += tile_a[local_id.x][t_k] * tile_b[t_k][local_id.y];
        }

        workgroupBarrier();
    }

    let x = local_id.x + (group_x * size);
    let y = local_id.y + (group_y * size);
    if (x < n && y < m) {
        let output_index = pointing(x, y, stride_output);
        let heap_index = pointer_output.x + output_index;
        heap[heap_index] = acc;
    }
}

fn pointing(x:u32, y:u32, stride:vec2<u32>) -> u32{
    return x * stride.x + y * stride.y;
}