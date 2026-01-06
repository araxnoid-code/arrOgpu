use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};

fn main() {
    let module = ArrOgpuModule::default();

    let array = module
        .array_from_vector(&[0., 2., 9., 8., 20., 1., 30., 29.], &[2, 4])
        .unwrap();
    println!("{}", array);

    let scalar = module.array_from_vector(&[7.], &[1]).unwrap();
    let result = module.powf(&array.view().unwrap(), &scalar).unwrap();
    println!("{}", result);

    println!(
        "{:?}",
        array
            .get_heap()
            .iter()
            .map(|v| v.powf(7.))
            .collect::<Vec<f32>>()
    );
}
