// heap
@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;

// slicing list
@group(1) @binding(0)
var<storage, read> slicing_list_start: array<u32>;
@group(1) @binding(1)
var<storage, read> slicing_list_end: array<u32>;

// loop
@group(1) @binding(2)
var<storage, read> _loops: array<u32>;

// stride
@group(1) @binding(3)
var<storage, read> array_stride: array<u32>;

// array pointer
@group(1) @binding(4)
var<uniform> array_pointer: vec2<u32>;

// output pointer
@group(1) @binding(5)
var<uniform> output_pointer: vec2<u32>;

// unit
@group(1) @binding(6)
var<uniform> total_unit: u32;

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) global_id:vec3<u32>){
    if global_id.x < total_unit{
        var start = 0u;
        var end = 0u;
        var iter = 0u;
        let slice_len = arrayLength(&slicing_list_start);

        for (var i = 0u; i < slice_len; i++){
            if i + 1u == slice_len{
                let range_start = slicing_list_start[i] * array_stride[i];
                let range_end =  slicing_list_end[i] * array_stride[i];
                let len = range_end - range_start;

                start += range_start;
                end += start + len;
                iter = len;
            } else {
                // index = floor(i/loop) - range * floor(i/(loop * range))
                // i: iterasi => actually "i"
                // range: range, that all => slice_end - slice_start
                // loop: stack 
                let range = slicing_list_end[i] - slicing_list_start[i];

                let _loop = _loops[i];
                let _index = (global_id.x / _loop) % range;

                let index = _index +  slicing_list_start[i];
                start += array_stride[i] * index;
            }
        }

        let i = global_id.y;
        if i < iter{
            let idx = array_pointer.x + start + i;
            let pointer_index = output_pointer.x + i + iter * global_id.x;
            heap[pointer_index] = heap[idx];
        }
    }
}

