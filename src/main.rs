use arr_o_gpu::{ ArrOgpuModule, GpuArray };

fn main() {
    let module = ArrOgpuModule::default();

    let array_a = module
        .array_from_vector(&(1..13).map(|v| v as f32).collect::<Vec<f32>>(), &[3, 4])
        .unwrap();

    let array_b = module
        .array_from_vector(&(1..13).map(|v| v as f32).collect::<Vec<f32>>(), &[4, 3])
        .unwrap();

    let out = module.matmul_2d(array_a, array_b).unwrap();

    println!("{}", out);
}
