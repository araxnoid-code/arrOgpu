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

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id:vec3<u32>){
    let length = pointer_o.y - pointer_o.x;
    if global_id.x < length{
        let index_a = pointer_a.x + indexing(global_id.x);
        let index_b = pointer_b.x + skalar_indexing();
        heap[pointer_o.x + global_id.x] = heap[index_a] - heap[index_b];
    }
}

fn indexing(x:u32) -> u32{
    var index = offset_a;
    let length = arrayLength(&shape_a);
    for (var i = 0u; i < length; i++){
        let permute = (x / iters_a[i]) % shape_a[i];
        index += (permute * stride_a[i]);
    }
    return index;
}


fn skalar_indexing() -> u32{
    let x = 0u;
    var index = offset_b;
    let length = arrayLength(&shape_b);
    for (var i = 0u; i < length; i++){
        let permute = (x / iters_b[i]) % shape_b[i];
        index += (permute *  stride_b[i]);
    }
    return index;
}