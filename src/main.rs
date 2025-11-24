use std::time::UNIX_EPOCH;

use arr_o_gpu::ArrOgpuModule;
use ndarray::{Array, ArrayD, Slice, *};

fn main() {
    let data = (0..4096)
        .into_iter()
        .map(|v| v as f32)
        .collect::<Vec<f32>>();
    let array_a = Array2::from_shape_vec([32, 128], data.clone()).unwrap();
    let array_b = Array2::from_shape_vec([128, 32], data.clone()).unwrap();

    let to_vec_numpy = array_a.dot(&array_b).flatten().to_vec();

    println!("==============");

    let module = ArrOgpuModule::default();

    let array_a = module.array_from_vector(&data, &[32, 128]).unwrap();
    // println!("{}", array_a);
    let array_b = module.array_from_vector(&data, &[128, 32]).unwrap();
    // println!("{}", array_b);

    let to_vec_my = module.matmul_nd(&array_a, &array_b).unwrap().get_heap();

    println!("{:?}", to_vec_my == to_vec_numpy)
}
