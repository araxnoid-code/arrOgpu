use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};

fn main() {
    let module = ArrOgpuModule::default();

    let array = module
        .array_from_vector(&[-1., 2., 3., 4., -10., 0., -89., 90., 33., -23.], &[5, 2])
        .unwrap();
    let array = array.view();
    let array = module.permute(&array, &[1, 0]).unwrap();
    // println!("{}", array);
    // let array_view = array.view();
    // let array_view = module.index(&array_view, &[1]).unwrap();

    let abs = module.abs(&array).unwrap();
    // module.abs(&array_view);

    // let sum = module.sum(&array).unwrap();

    println!("{abs}");
}
