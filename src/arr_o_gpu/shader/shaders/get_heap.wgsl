@group(0) @binding(0)
var <storage, read_write> heap: array<f32>;

@group(1) @binding(0)
var <storage, read_write> copy: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id:vec3<u32>){
    let len = arrayLength(&heap);

    for (var i:u32 = 0; i < len; i++){
        copy[i] = heap[i];
    }
}