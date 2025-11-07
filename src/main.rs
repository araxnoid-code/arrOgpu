use std::time::SystemTime;

use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    let array = module.array_from_vector(&[1.0, 2.0, 3.0], &[3, 1]).unwrap();
    module.broadcasting(&array, &[2, 3]).unwrap();
}
