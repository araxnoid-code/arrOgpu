use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    let array_a = module
        .array_from_vector(&(0..36).map(|v| v as f32).collect::<Vec<f32>>(), &[2, 2, 3, 3])
        .unwrap();

    // let array_b = module.array_from_vector(&[11.0, 12.0, 13.0, 14.0, 15.0, 29.0], &[6]);

    println!("{}", array_a);
}
