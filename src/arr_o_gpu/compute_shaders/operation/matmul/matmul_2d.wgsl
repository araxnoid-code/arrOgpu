// override
override LEN_HEAP_MINUS_ONE: u32;
override WORKGROUP_SIZE: u32 = 16;

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
// // heap
@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;
// // execute_args
@group(0) @binding(1)
var<uniform> execute_args: array<ArrayMetadata, 3>;

// cache
// akan disesuaikan dengan WORKGROUP_SIZE saat sebelum kompilasi
var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn main(
@builtin (local_invocation_id) local_id: vec3<u32>,
@builtin (global_invocation_id) global_id: vec3<u32>,
@builtin (workgroup_id) group_id: vec3<u32>,
){
    let m = execute_args[0].shape[0][0];
    let k = execute_args[0].shape[0][1];
    let n = execute_args[1].shape[0][1];

    let iter = (k + 15) / WORKGROUP_SIZE;
    var acc = 0.;
    for (var i = 0u; i < iter; i++){
        // A
        let row_a = global_id.x;
        let col_a = local_id.y + WORKGROUP_SIZE * i;

        var a_value = select(0., heap[indexing_a(row_a, col_a)], row_a < m && col_a < k);
        tile_a[local_id.x][local_id.y] = a_value;

        // B
        let row_b = local_id.x + WORKGROUP_SIZE * i;
        let col_b = global_id.y;

        var b_value = select(0., heap[indexing_b(row_b, col_b)], row_b < k && col_b < n);
        tile_b[local_id.x][local_id.y] = b_value;

        workgroupBarrier();

        for (var ii = 0u; ii < WORKGROUP_SIZE; ii++){
            acc += tile_a[local_id.x][ii] * tile_b[ii][local_id.y];
        }

        workgroupBarrier();
    }

    if global_id.x >= m || global_id.y >= n{
        return;
    }

    heap[indexing_out(global_id.x, global_id.y)] = acc;
}

fn indexing_a(row:u32, col:u32)-> u32{
    let arr = execute_args[0];
    let stride = arr.stride[0];
    let index = arr.pointer.x + arr.offset + row * stride[0] + col * stride[1];
    return select(0, index, index < LEN_HEAP_MINUS_ONE);
}

fn indexing_b(row:u32, col:u32)-> u32{
    let arr = execute_args[1];
    let stride = arr.stride[0];
    let index = arr.pointer.x + arr.offset + row * stride[0] + col * stride[1];
    return select(0, index, index < LEN_HEAP_MINUS_ONE);
}

fn indexing_out(row: u32, col:u32)-> u32{
    let arr = execute_args[2];
    let stride = arr.stride[0];
    let index = arr.pointer.x + arr.offset + row * stride[0] + col * stride[1];
    return index;
}
