use arr_o_gpu::{ArrOgpuModule, HeapSize};

fn main() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        heap_size: HeapSize::Item(100000),
        wgpu: arr_o_gpu::WgpuInit::ManualInit(arr_o_gpu::ManualInit {
            power: arr_o_gpu::Power::HighPerformance,
            memory: arr_o_gpu::Memory::Performance,
        }),
    })
    .unwrap();

    let shape = [10];
    let data = vec![1.; 10];
    let array_a = module.array_from_vector(&data, &shape).unwrap();
    let array_b = module.array_from_vector(&data, &shape).unwrap();
    // println!("{:?}", array_b.pointer_to_arr());

    let dot = module.dot_product_optimize(&array_a, &array_b).unwrap();
    println!("{:?}", &module.get_heap()[21..30]);
}
