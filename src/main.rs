use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    let array = module
        .array_from_vector(&[1., 2., 3., 4., 5., 6.], &[2, 3])
        .unwrap();

    let result = array.add(&1.).unwrap();

    println!("{}", result);
    // [
    //  [2.0, 3.0, 4.0]
    //  [5.0, 6.0, 7.0]
    // ]
}
