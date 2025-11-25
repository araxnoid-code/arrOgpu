use std::ops::{Range, RangeFull};

use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, SlicingRangeTrait};

fn main() {
    let module = ArrOgpuModule::default();
    let array = ArangeArray::arange(10..20)
        .map(|v| v)
        .step_by(1)
        .to_GpuArray_with_shape(&[5, 2], &module)
        .unwrap();
}
