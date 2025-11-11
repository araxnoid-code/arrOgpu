use std::time::SystemTime;

use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();
    let array = module.array_from_vector(&[], &[]).unwrap();

    let broadcast = module.broadcasting(&array, &[]);
}
