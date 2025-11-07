// heap
@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;

@group(1) @binding(0)
var<uniform> thread_limit: u32;

@group(1) @binding(1)
var<uniform> stride_target: u32;

@group(1) @binding(2)
var<uniform> stride_output: u32;

@group(1) @binding(3)
var<uniform> extend_count: u32;

@group(2) @binding(0)
var<uniform> arr_pointer: vec2<u32>;

@group(2) @binding(1)
var<uniform> out_pointer: vec2<u32>;

@compute @workgroup_size(16, 16, 1) // (global_loop, stride_target, 1)
fn main(
    @builtin(global_invocation_id) global_id:vec3<u32>
){
    if global_id.x < thread_limit{
        let start = arr_pointer.x + stride_target * global_id.x;
        let end = start + stride_target;

        if global_id.y < stride_target{
            for(var i = 0u; i < extend_count; i++){
                let heap_element_index = start + global_id.y;
                let heap_place_index = ((global_id.y + stride_target * i) + stride_output * global_id.x) + out_pointer.x;

                heap[heap_place_index] = heap[heap_element_index];
            }
        }
        
    }
}