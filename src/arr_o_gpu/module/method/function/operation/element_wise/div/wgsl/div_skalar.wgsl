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

// skalar 
@group(2) @binding(0)
var<uniform> skalar: f32;

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
        heap[pointer_o.x + global_id.x] = heap[index_a] / skalar;
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
