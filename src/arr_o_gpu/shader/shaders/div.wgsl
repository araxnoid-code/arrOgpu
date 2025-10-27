// heap
@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;

// array a
@group(1) @binding(0)
var<uniform> pointer_a: vec2<u32>;

// array b
@group(1) @binding(1)
var<uniform> pointer_b: vec2<u32>;

// output
@group(1) @binding(2)
var<uniform> pointer_out: vec2<u32>;


@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id:vec3<u32>){
    let index_heap =  pointer_out.x + global_id.x;
    if (index_heap < pointer_out.y){
        let index_a = pointer_a.x + global_id.x;
        let v_a = heap[index_a];

        let index_b = pointer_b.x + global_id.x;
        let v_b = heap[index_b];

        heap[index_heap] = v_a / v_b;
    }
}