use arr_o_gpu::{ ArrOgpuModule, GpuArray };

fn main() {
    let module = ArrOgpuModule::default();

    let array_a = module
        .array_from_vector(&(1..5).map(|v| v as f32).collect::<Vec<f32>>(), &[2, 2])
        .unwrap();

    let array_b = module
        .array_from_vector(&(1..5).map(|v| v as f32).collect::<Vec<f32>>(), &[2, 2])
        .unwrap();

    module.matmul_2d(array_a, array_b).unwrap();

    // println!("{:?}", module.get_heap());
}
