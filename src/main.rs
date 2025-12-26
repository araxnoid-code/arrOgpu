use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    let array = module
        .array_from_vector(&[1., 2., 3., 4.], &[2, 2])
        .unwrap();
    let array = module.permute(&array, &[1, 0]).unwrap();

    let power = module.array_from_vector(&[2.], &[1]).unwrap();

    let pow = module.pow(&array, &power).unwrap();
    println!("{}", pow);
}
