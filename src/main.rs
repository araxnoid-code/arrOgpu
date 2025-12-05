use std::{ ops::{ Range, RangeFull }, sync::mpsc, time::UNIX_EPOCH };

use arr_o_gpu::{ ArangeArray, ArrOgpuModule, FlatteTrait, r };
use ndarray::{ Array1, Array2 };

fn main() {
    let len = 54;
    let data = (0..len)
        .into_iter()
        .map(|v| v as f32)
        .collect::<Vec<f32>>();

    let module = ArrOgpuModule::default();

    let array_a = module.array_from_vector(&data, &[2, 3, 3, 3]).unwrap();
    println!("{}", array_a);
    println!("{}", module.slicing(&array_a, &[r(0..2), r(0..3), r(1..3), r(1..3)]).unwrap());
}
