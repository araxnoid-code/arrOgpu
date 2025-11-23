use std::ops::{Range, RangeFull};

use arr_o_gpu::{ArangeArray, ArrOgpuModule, SlicingRangeTrait};

fn main() {
    let module = ArrOgpuModule::default();
    let array = module.arange(10..20).map(|v| v + 10.).step_by(2);
}
