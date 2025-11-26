use std::ops::{Range, RangeFull};

use arr_o_gpu::{ArrOgpuModule, FlatteTrait};

fn main() {
    let module = ArrOgpuModule::default();

    let array = module
        .new([[0., 0., 0.], [1., 1., 1.], [1., 1., 1.]])
        .unwrap();
    println!("{}", array);
}
