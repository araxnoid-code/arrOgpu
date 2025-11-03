use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    let data = (0..2 * 2 * 3 * 2).map(|a| a as f32).collect::<Vec<f32>>();
    let array_a = module.array_from_vector(&data, &[2, 2, 3, 2]).unwrap();
    println!("{}", array_a);

    let data = (0..2 * 2 * 3 * 2).map(|a| a as f32).collect::<Vec<f32>>();
    let array_b = module.array_from_vector(&data, &[2, 2, 2, 3]).unwrap();
    println!("{}", array_b);

    module.matmul_nd(&array_a, &array_b).unwrap();

    // println!("{:?}", module.get_heap());
}
