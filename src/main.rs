use std::ops::{Range, RangeFull};

use arr_o_gpu::FlatteTrait;

fn main() {
    // let module = ArrOgpuModule::default();

    let data = vec![vec![1f32], vec![2f32], vec![3f32], vec![4f32]];
    let flat = data.flatten();
    println!("{:?}", flat)

    // module.new(&[10.]);
    // let array = ArangeArray::arange(10..20)
    //     .map(|v| v)
    //     .step_by(1)
    //     .to_GpuArray_with_shape(&[5, 2], &module)
    //     .unwrap();
}
