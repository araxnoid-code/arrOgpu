@group(0) @binding(0)
var <storage, read_write> heap: array<f32>;

@group(1) @binding(0)
// vector
var <storage, read> init_data: array<f32>;

// pointer
@group(1) @binding(1)
var <storage, read> init_pointer: vec2<u32>;

@compute @workgroup_size(1)
fn array_init(@builtin(global_invocation_id) global_id: vec3<u32>){
    let id = global_id.x; // 0..dispatch
    let idx = id + init_pointer[0];

    if (idx < init_pointer[0]){
        heap[idx] = init_data[id];
    }
}