// heap
@group(0) @binding(0)
var<storage, read_write> heap: array<f32>;

// array a
// // pointer
var<uniform> pointer_a: vec2<u32>;

// array b
// // pointer
var<uniform> pointer_b: vec2<u32>;

// k (length of array)
@group(1) @binding(0)
var<uniform> k: u32;

// output
// // pointer
@group(1) @binding(1)
var<uniform> pointer_output: vec2<u32>;

// workgroup
// // cache to store result of paralel tree
var<workgroup> cache_tree: array<f32, 256>;
// // cache to store result multiple operation
// var<workgroup> cache_mul: array<f32, 256>

// Description
// --
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id:vec3<u32>){
    // ceil(a/b) = (a + b - 1)/b
    let iter = (k + 256u - 1u) / 256u;

    // iteration
    // iterate to get the entire array from start to finish
    var acc = 0.;
    for (var i = 0u; i < iter; i++){

        // multiple
        var jump = 256u * i;
        if (jump + global_id.x < k){
            var a_idx = pointer_a.x + global_id.x + jump;
            var b_idx = pointer_b.x + global_id.x + jump;
            var cache_idx = global_id.x;
            cache_tree[cache_idx] = heap[a_idx] * heap[b_idx];
        }
        workgroupBarrier();

        var chace_length = k - 256 * i;
        var total_unit = (chace_length + 2 - 1) / 2;
        while true {
            if global_id.x < total_unit{
                let start = global_id.x + global_id.x;
                let end = start + 1;

                var sum = 0.;
                if (end < chace_length){
                    sum = cache_tree[start] + cache_tree[end];
                } else {
                    sum = cache_tree[start];
                }

                cache_tree[global_id.x] = sum;

                if (total_unit == 1){break;}

                chace_length = total_unit;
                total_unit = (total_unit + 2 - 1) / 2;

            } else {
                break;
            }
        }
        workgroupBarrier();

        acc = acc + cache_tree[0];
    }

    heap[pointer_output.x] = acc;
}