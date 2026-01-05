use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};

fn main() {
    let module = ArrOgpuModule::default();

    let array = module
        .array_from_vector(&[0., 1., -9., -8., 20., -1., 30., -29.], &[2, 4])
        .unwrap();
    println!("{}", array);

    let result = module.abs(&array.view().unwrap()).unwrap();
    println!("{}", result);
}
