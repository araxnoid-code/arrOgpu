// heap
@group(0) @binding(0)
var <storage, read_write> heap:array<f32>;

@group(1) @binding(0)
var <storage, read_write> output: array<f32>;

@group(1) @binding(1)
var <storage, read> pointer_output: array<f32>;

@group(1) @binding(2)
var <storage, read> stride_output: vec2<u32>;


@compute @workgroup_size(1)
fn main(){}