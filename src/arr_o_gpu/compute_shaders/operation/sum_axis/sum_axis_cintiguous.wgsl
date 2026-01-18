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

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin (global_invocation_id) global_id: vec3<u32>
){
    if global_id.x < execute_args[1].len{
        let sum_len = get_sum_len();
        if global_id.y < sum_len{

            let in_axis_shape = get_in_axis_shape();
            let in_axis_stride = get_in_axis_stride(in_axis_shape);
            let mark_stride = get_mark_stride();

            var index = execute_args[0].offset;

            for (var i = 0u; i < 8; i++){
                let axis = static_cache[i >> 2][i & 3];
                if axis != 0 || i == 0{
                    let permute = (global_id.y / in_axis_stride[i]) % in_axis_shape[i];
                    index += permute * execute_args[0].stride[axis >> 2][axis & 3];
                }
            }

            for (var i = 0u; i < execute_args[1].dim; i++){
                let idx0 = i >> 2;
                let idx1 = i % 3;
                let permute = (global_id.x / execute_args[1].o_stride[idx0][idx1]) * execute_args[1].shape[idx0][idx1];
                index += permute * mark_stride[i];
            }
        }
    }
}

fn get_in_axis_shape() -> array<u32, 9> {
    var in_axis_shape = array<u32, 9>(0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (var i = 0u; i < execute_args[0].dim; i++){
        in_axis_shape[i] = 1;
    }

    for (var i = 0u; i < 8; i++){
        let axis = static_cache[i >> 2][i & 3];
        let zero = u32(axis != 0 || i == 0);

        let index = select(8, axis, axis != 0 || i == 0);
        in_axis_shape[index] = execute_args[0].shape[axis >> 2][axis & 3];
    }

    return in_axis_shape;
}

fn get_in_axis_stride(in_axis_shape: array<u32, 9>) -> array<u32, 8>{
    var in_axis_sride = array<u32, 8>(1, 1, 1, 1, 1, 1, 1, 1);
    for (var i = 0u; i < execute_args[0].dim; i++){
        for (var ii = i + 1; ii < execute_args[0].dim; ii++){
            in_axis_sride[i] *= execute_args[0].shape[ii >> 2][ii & 3];
        }
    }

    return in_axis_sride;
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
        if execute_args[0].shape[i >> 2][i & 3] != 0{
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
