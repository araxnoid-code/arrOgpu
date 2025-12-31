use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    let array = module.array_from_vector(&[1.0], &[1]).unwrap();
    println!("{}", array);
}
