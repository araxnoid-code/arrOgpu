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
@group(2) @binding(0)
var<uniform> pointer_o: vec2<u32>;

// // shape
@group(2) @binding(1)
var<storage, read> shape_o: array<u32>;

// // iters
@group(2) @binding(2)
var<storage, read> iters_o: array<u32>;

// // stride
@group(2) @binding(3)
var<storage, read> stride_o: array<u32>;

// // offset
@group(2) @binding(4)
var<uniform> offset_o: u32;

// power
// // pointer
@group(3) @binding(0)
var<uniform> pointer_p: vec2<u32>;

// // shape
@group(3) @binding(1)
var<storage, read> shape_p: array<u32>;

// // iters
@group(3) @binding(2)
var<storage, read> iters_p: array<u32>;

// // stride
@group(3) @binding(3)
var<storage, read> strid_p: array<u32>;

// // offset
@group(3) @binding(4)
var<uniform> offset_p: u32;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
){
    let len = pointer.y - pointer.x;
    if global_id.x < len{
        let power =  heap[ offset_p + pointer_p.x ];

        let x = heap[ pointer.x + global_id.x ];
        var result = pow(abs(x), power);

        let is_odd = f32(u32(power) % 2);
        let negatif_counter = mix(1., sign(x), is_odd);
        result *= negatif_counter;

        heap[pointer_o.x + global_id.x] = result;

    }
}
