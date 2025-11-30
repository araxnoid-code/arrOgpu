use std::{ ops::{ Range, RangeFull }, sync::mpsc, time::UNIX_EPOCH };

use arr_o_gpu::{ ArangeArray, ArrOgpuModule, FlatteTrait };
use ndarray::{ Array1, Array2 };

fn main() {
    let len = 6;
    let data = (0..len)
        .into_iter()
        .map(|v| v as f32)
        .collect::<Vec<f32>>();

    let module = ArrOgpuModule::default();

    let array_a = module.array_from_vector(&data, &[1, 2, 3]).unwrap();
    println!("{}", module.broadcasting(&array_a, &[2, 2, 3]).unwrap());
}
