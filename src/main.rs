use std::time::SystemTime;

use arr_o_gpu::{ ArrOgpuModule, r };

fn main() {
    let module = ArrOgpuModule::default();

    let data = (0..6).map(|v| v as f32).collect::<Vec<f32>>();
    let array = module.array_from_vector(&data, &[2, 3]).unwrap();

    module.slicing(&array, &mut [r(..1), r(..)]).unwrap();
}
