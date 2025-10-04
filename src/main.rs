use arr_o_gpu::*;
use arr_o_gpu::{ ArrOgpuModule, FlatteTrait };

struct MyOpp {
    test: String,
}

fn main() {
    let mut module = ArrOgpuModule::default();
    module.array_from_vector(&vec![100.0; 99], &[1, 64]);
    module.get_heap();
}
