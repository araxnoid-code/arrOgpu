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

@group(1) @binding(4)
var<uniform> arr_pointer: vec2<u32>;

@group(1) @binding(5)
var<uniform> out_pointer: vec2<u32>;

// workgroup
var<workgroup> cache: array<f32, 256>;

@compute @workgroup_size(16, 16, 1) // (global_loop, stride_target, 1)
fn main(
    @builtin(global_invocation_id) global_id:vec3<u32>,
    @builtin(local_invocation_id) local_id:vec3<u32>
){
    if global_id.x < thread_limit{
        let start = arr_pointer.x + stride_target * global_id.x;

        let cache_row_size = 16u;
        let cache_colm_size = 16u;

        let row = local_id.x * cache_row_size;
        let cache_index = row + local_id.y;

        if local_id.y < stride_target{
            let heap_index = start + local_id.y;
            cache[cache_index] = heap[heap_index];
        }

        workgroupBarrier();

        if global_id.y < stride_output{
            let index_element = global_id.y - u32(floor(f32(global_id.y) / f32(stride_target))) * stride_target;
            let index_heap = global_id.y + out_pointer.x + stride_output * global_id.x;
            heap[index_heap] = cache[row + index_element];
        }
    }
}