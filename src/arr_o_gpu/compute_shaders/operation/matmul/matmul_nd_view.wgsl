// workgroup
// will be adjusted to WORKGROUP_SIZE before compilation
var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

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
// heap
@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;
// execute_args
@group(0) @binding(1)
var<uniform> execute_args: array<ArrayMetadata, 3>;

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn main(
    @builtin (local_invocation_id) local_id: vec3<u32>,
    @builtin (global_invocation_id) global_id: vec3<u32>,
    @builtin (workgroup_id) work_id: vec3<u32>,
){
    let array_a = execute_args[0];
    let array_b = execute_args[1];
    let dim = array_a.dim;

    let m = array_a.shape[(dim - 2) >> 2][(dim - 2) & 3];
    let k = array_a.shape[(dim - 1) >> 2][(dim - 1) & 3];
    let n = array_b.shape[(dim - 1) >> 2][(dim - 1) & 3];

    var batch_stride_a = 0u;
    for(var i = 0u; i < array_a.dim - 2; i++){
        let idx0 = i >> 2;
        let idx1 = i & 3;
        let permute = ((m * k * global_id.z) / array_a.o_stride[idx0][idx1]) % array_a.shape[idx0][idx1];
        batch_stride_a += permute * array_a.stride[idx0][idx1];
    }

    var batch_stride_b = 0u;
    for(var i = 0u; i < array_a.dim - 2; i++){
        let idx0 = i >> 2;
        let idx1 = i & 3;
        let permute = ((k * n * global_id.z) / array_b.o_stride[idx0][idx1]) % array_b.shape[idx0][idx1];
        batch_stride_b += permute * array_b.stride[idx0][idx1];
    }

    let iteration = (k + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    var acc = 0.;
    for (var i = 0u; i < iteration; i++){
        let row_a = global_id.x;
        let coll_a = local_id.y + WORKGROUP_SIZE * i;
        let value_a = select(0., heap[indexing_a(row_a, coll_a, batch_stride_a)], row_a < m && coll_a < k);
        tile_a[local_id.x][local_id.y] = value_a;

        let row_b = local_id.x + WORKGROUP_SIZE * i;
        let coll_b = global_id.y;
        let value_b = select(0., heap[indexing_b(row_b, coll_b, batch_stride_b)], row_b < k && coll_b < n);
        tile_b[local_id.x][local_id.y] = value_b;

        workgroupBarrier();
        for (var ii = 0u; ii < WORKGROUP_SIZE; ii++){
           acc += tile_a[local_id.x][ii] * tile_b[ii][local_id.y];
        }
        workgroupBarrier();
    }

    if global_id.x >= m || global_id.y >= n{
        return;
    }

    let index = indexing_o(global_id.x, global_id.y, global_id.z);
    heap[index] = acc;

}

fn indexing_a(row: u32, coll: u32, batch_stride: u32) -> u32{
    let arr = execute_args[0];
    let index = arr.dim - 1;
    let row_stride = arr.stride[(index - 1) >> 2][(index - 1) & 3];
    let coll_stride = arr.stride[index >> 2][index & 3];

    let out = arr.offset + arr.pointer.x + row * row_stride + coll * coll_stride + batch_stride;
    return select(0, out, out < LEN_HEAP_MINUS_ONE);
}

fn indexing_b(row: u32, coll: u32, batch_stride:u32) -> u32{
    let arr = execute_args[1];
    let index = arr.dim - 1;
    let row_stride = arr.stride[(index - 1) >> 2][(index - 1) & 3];
    let coll_stride = arr.stride[index >> 2][index & 3];

    let out = arr.offset + arr.pointer.x + row * row_stride + coll * coll_stride + batch_stride;
    return select(0, out, out < LEN_HEAP_MINUS_ONE);
}

fn indexing_o(row: u32, coll: u32, z:u32) -> u32{
    let arr = execute_args[2];
    let index = arr.dim - 1;
    let row_stride = arr.stride[(index - 1) >> 2][(index - 1) & 3];
    let coll_stride = arr.stride[index >> 2][index & 3];
    let other_stride = arr.stride[(index - 2) >> 2][(index - 2) & 3];

    return arr.offset + arr.pointer.x + row * row_stride + coll * coll_stride + other_stride * z;
}
