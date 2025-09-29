@group(0) @binding(0)
var <storage, read_write> allocator: array<f32>;

@compute @workgroup_size(1)
fn init(@builtin(global_invocation_id) global_id:vec3<u32>){
    var data = allocator[0];
}