use arr_o_gpu::{ ArrOgpuModule, GpuArray };

fn main() {
    let module = ArrOgpuModule::default();

    {
        let array_a = module
            .array_from_vector(&(0..18).map(|v| v as f32).collect::<Vec<f32>>(), &[2, 3, 3])
            .unwrap();

        let indexing = array_a.index(&[1, 1]).unwrap();
    }

    println!("{:?}", module.allocator_read().monanagement.range_space);
}
