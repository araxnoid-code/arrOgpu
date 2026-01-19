// INIT
struct ArrayMetadata{
	pointer: vec2<u32>, // 8
	len: u32, // 4
	offset: u32, // 4
	dim: u32, // 4
	padding0: u32, // 4
	padding1: u32, // 4
	padding2: u32, // 4
	shape: array<vec4<u32>, 2>, // 32
	stride: array<vec4<u32>, 2>, // 32
	o_stride: array<vec4<u32>, 2>, // 32
	m_n_shape: array<vec4<u32>, 2>, // 32
	m_n_origin_stride: array<vec4<u32>, 2>, // 32
	padding3: array<vec4<u32>, 4>,
}

// MODULE
// heap
@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;
// Execute args
@group(0) @binding(1)
var<uniform> execute_args: array<ArrayMetadata, 3>;
// static_cache
@group(0) @binding(2)
var<uniform> static_cache: array<vec4<u32>, 2>;

// Reduction
struct Counter{
    value: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
    padding3: array<vec4<u32>, 15>
}

@group(1) @binding(0)
var<storage, read_write> reduction_heap: array<f32>;

@group(1) @binding(1)
var<uniform> reduction_counter: Counter;

@group(1) @binding(2)
var<storage, read_write> reduction_len: Counter;

// cache
var<workgroup> cache: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin (global_invocation_id) global_id: vec3<u32>,
    @builtin (local_invocation_id) local_id: vec3<u32>,
    @builtin (workgroup_id) work_id: vec3<u32>,
){
    let len = select(reduction_len.value, execute_args[1].len, reduction_counter.value == 0);
    let total_thread_y = 16u;
    var reduction_len = total_thread_y;
    if reduction_len > (work_id.y + 1) * 16{
        reduction_len = reduction_len - work_id.y * 16;
    }

    let sum_len = get_sum_len();
    if global_id.x < len && reduction_counter.value == 0{
        if global_id.y < sum_len{

            let in_axis_shape = get_in_axis_shape();
            let in_axis_stride = get_in_axis_stride(in_axis_shape);
            let mark_stride = get_mark_stride();

            var index = execute_args[0].pointer.x + execute_args[0].offset;
            for (var i = 0u; i < 8; i++){
                let axis = static_cache[i >> 2][i & 3];
                if axis != 0 || i == 0{
                    let permute = (global_id.y / in_axis_stride[axis]) % in_axis_shape[axis];
                    index += permute * execute_args[0].stride[axis >> 2][axis & 3];
                }
            }

            for (var i = 0u; i < execute_args[1].dim; i++){
                let permute = (global_id.x / execute_args[1].o_stride[i >> 2][i & 3]) % execute_args[1].shape[i >> 2][i & 3];
                index += permute * mark_stride[i];
            }

            cache[local_id.x][local_id.y] = heap[index];
        }
    } else if global_id.x < len && global_id.y < reduction_len {
        let index = len * global_id.x + global_id.y;
        cache[local_id.x][local_id.y] = reduction_heap[index];
    }
    workgroupBarrier();



    var stride = 1u;
    while (stride < total_thread_y){
        let start = local_id.y * 2 * stride;
        let index = select(0 , start + stride, start + stride < reduction_len);
        cache[local_id.x][start] += select(0, cache[local_id.x][index], start + stride < reduction_len);
        stride <<= 1;
        workgroupBarrier();
    };

    if global_id.x >= len || global_id.y != 0{
        return;
    }

    if (len + 15) >> 4 == 1{
        heap[global_id.x + execute_args[1].pointer.x] = cache[local_id.x][0];
    } else {
        let index = len * global_id.x + work_id.y;
        reduction_heap[index] = cache[local_id.x][0];
    }

}

fn get_in_axis_shape() -> array<u32, 9> {
    var in_axis_shape = array<u32, 9>(1, 1, 1, 1, 1, 1, 1, 1, 0);

    for (var i = 0u; i < 8; i++){
        let axis = static_cache[i >> 2][i & 3];
        let index = select(8, axis, axis != 0 || i == 0);
        in_axis_shape[index] = execute_args[0].shape[axis >> 2][axis & 3];
    }

    return in_axis_shape;
}

fn get_in_axis_stride(in_axis_shape: array<u32, 9>) -> array<u32, 8>{
    var in_axis_stride = array<u32, 8>(1, 1, 1, 1, 1, 1, 1, 1);
    for (var i = 0u; i < execute_args[0].dim; i++){
        for (var ii = i + 1; ii < execute_args[0].dim; ii++){
            in_axis_stride[i] *= in_axis_shape[ii];
        }
    }
    return in_axis_stride;
}

fn get_mark_stride() -> array<u32, 8>{
    var mark_stride = array<u32, 8>(0, 0, 0, 0, 0, 0, 0, 0);

    var mark_shape = execute_args[0].shape;
    for (var i = 0u; i < 8; i++){
        let axis = static_cache[i >> 2][i & 3];
        if axis != 0 || i == 0{
            mark_shape[axis >> 2][axis & 3] = 0;
        }
    }

    var skip = 0u;
    for (var i = 0u; i < execute_args[0].dim; i++){
        if mark_shape[i >> 2][i & 3] != 0{
            let stride = execute_args[0].stride[i >> 2][i & 3];
            mark_stride[i - skip] = stride;
        } else {
            skip += 1;
        }
    }

    return mark_stride;
}

fn get_sum_len() -> u32{
    var len = 1u;
    for (var i = 0u; i < 8; i++){
        let axis = static_cache[i >> 2][i & 3];
        let zero = u32(axis != 0 || i == 0);
        len *=  1 * (1 - zero) + execute_args[0].shape[axis >> 2][axis & 3] * zero;
    }
    return len;
}
