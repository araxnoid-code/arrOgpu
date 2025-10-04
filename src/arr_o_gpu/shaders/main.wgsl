// @group(0) @binding(0)
// var <storage, read_write> allocator: array<f32>;

// @compute @workgroup_size(1)
// fn init(@builtin(global_invocation_id) global_id:vec3<u32>){
//     var data = allocator[0];
// }

// // initialization
// // array_from_vector
// @group(1) @binding(0)
// // vector
// var <storage, read> init_data: array<f32>;

// // pointer
// @group(1) @binding(1)
// var <storage, read> init_pointer: vec2<u32>;

// // shape
// @group(1) @binding(2)
// var <storage, read> init_shape: array<u32>;

// @compute @workgroup_size(1)
// fn array_init(@builtin(global_invocation_id) global_id: vec3<u32>){

// }