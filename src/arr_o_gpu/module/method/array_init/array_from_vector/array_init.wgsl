@group(0) @binding(0)
var <storage, read_write> heap: array<f32>;

@group(1) @binding(0)
// vector
var <storage, read> init_data: array<f32>;

// pointer
@group(1) @binding(1)
var <uniform> init_pointer: vec2<u32>;

@compute @workgroup_size(256)
fn array_init(@builtin(global_invocation_id) global_id: vec3<u32>){
    let id = global_id.x; // 0..dispatch
    let idx = id + init_pointer.x;

    if (idx < init_pointer.y){
        heap[idx] = init_data[id];
    }
}