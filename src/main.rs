use std::fmt::Debug;

use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r};

fn main() {
    let module = ArrOgpuModule::default();

    let array = module
        .array_from_vector(&[-1., 2., 3., 4., -10., 0., -89., 90., 33., -23.], &[5, 2])
        .unwrap();
    // let array = array.view();
    let array = module.permute(&array, &[1, 0]).unwrap();
    let array = module.reshape(&array, &[10]).unwrap();
    // let array = module.slicing(&array, &[r(2..5)]).unwrap();
    // println!("{}", array);
    // let array_view = array.view();
    // let array_view = module.index(&array_view, &[1]).unwrap();

    let abs = module.sqrt(&array).unwrap();
    // module.abs(&array_view);

    // let sum = module.sum(&array).unwrap();
}
