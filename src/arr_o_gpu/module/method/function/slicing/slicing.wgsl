// heap
@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;

// slicing list
@group(1) @binding(0)
var<uniform> slicing_list_start: array<u32>;
@group(1) @binding(1)
var<uniform> slicing_list_end: array<u32>;

// loop
@group(1) @binding(2)
var<uniform> _loops: array<f32>;

// stride
@group(1) @binding(3)
var<uniform> stride_list: array<u32>;

// array pointer
var<uniform> array_pointer: vec2<u32>;

// output pointer
var<uniform> output_pointer: vec2<u32>;

@compute @workgroup_size(16);
fn main(@builtin(global_invocation_id) global_id:vec3<u32>){
    var start = 0u;
    var end = 0u;
    var iter = 0u;
    let slice_len = length(&slicing_list_start);

    for var i = 0u; i < slice_len; i++{
        let range_start = slicing_list_start[i] * stride_list[i];
        let range_end =  slicing_list_end[i] * stride_list[i];
        let len = range_end - range_start;

        if i == slice_len{
            start += range_start;
            end += start + len;
            iter = len;
        } else {
            // index = floor(i/loop) - range * floor(i/(loop * range))
            // i: iterasi => actually "i"
            // range: range, that all => slice_end - slice_start
            // loop: stack 
            let range = slicing_list_end[i] - slicing_list_start[i];

            let _range = f32(range);
            let _loop = _loops[i];
            let _i = f32(i); 
            let _index = floor(_i / _loop) - _range * floor(_i/(_loop * _range));

            let index = u32(_index) +  slicing_list_start[i];
            start += stride_list[i] * index;
        }

        let start_pointer = array_pointer.0;
        for var i = 0u; i < iter; i++{
            let idx = start_pointer + start + i;

            let pointer_index = i + len * global_id.x;
            heap[pointer_index] = heap[idx];
        }
    }
}

