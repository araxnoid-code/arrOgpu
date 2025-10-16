// heap
@group(0) @binding(0)
var <storage, read_write> heap: array<f32>;

// output
@group(1) @binding(0)
var <storage, read_write> output: array<f32>;

// range
@group(1) @binding(1)
var <storage, read_write> range: array<f32>;

@compute @workgroup_size(128)
fn indexing(@builtin(global_invocation_id) global_id: vec3<u32>){
    
}