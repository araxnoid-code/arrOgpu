use std::{
    ops::{Range, RangeFull},
    sync::mpsc,
};

use arr_o_gpu::{ArrOgpuModule, FlatteTrait};

fn main() {
    let module = ArrOgpuModule::default();

    let array_a = module.new([1., 2., 3.]).unwrap();

    let array_b = module.new([1., 2., 3.]).unwrap();

    module.dot_product(&array_a, &array_b).unwrap();
}
