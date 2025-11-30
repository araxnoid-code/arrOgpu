// heap
@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;

// Thread Limit For X
@group(1) @binding(0)
var<uniform> thread_limit: u32;

// Stride Of Broadcast Target
@group(1) @binding(1)
var<uniform> stride_target: u32;

// Stride Of Broadcast Target Of Output Shape multiple by broadcast number
@group(1) @binding(2)
var<uniform> stride_output: u32;

// array
// // pointer
@group(1) @binding(3)
var<uniform> arr_pointer: vec2<u32>;

// output
// // pointer
@group(1) @binding(4)
var<uniform> out_pointer: vec2<u32>;

// Desscription
// The broadcast algorithm uses x as a grouping based on stride and y as a worker unit that will copy the data.
@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id:vec3<u32>
){
    if global_id.x < thread_limit{
        if global_id.y < stride_output{
            let start = arr_pointer.x + stride_target * global_id.x;

            // y will be linear, therefore a modulus is needed so that the range matches the range of elements in the array
            let index_element = global_id.y % stride_target;
            let index_element_on_heap = start + index_element;

            let index_heap = out_pointer.x + global_id.y + stride_output * global_id.x;
            heap[index_heap] = heap[index_element_on_heap];
        }
    }
}