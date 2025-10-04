use arr_o_gpu::*;
use arr_o_gpu::{ ArrOgpuModule, FlatteTrait };

struct MyOpp {
    test: String,
}

fn main() {
    let mut module = ArrOgpuModule::default();
    module.array_from_vector(&vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5]);
}
