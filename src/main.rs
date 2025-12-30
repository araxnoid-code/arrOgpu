use arr_o_gpu::{ArrOgpuModule, HeapSize};

fn main() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        heap_size: HeapSize::Item(1500000),
        wgpu: arr_o_gpu::WgpuInit::ManualInit(arr_o_gpu::ManualInit {
            power: arr_o_gpu::Power::HighPerformance,
            memory: arr_o_gpu::Memory::Performance,
        }),
    })
    .unwrap();

    let shape = [5000];
    let data = vec![2.; 5000];
    let array_a = module.array_from_vector(&data, &shape).unwrap();
    let array_b = module.array_from_vector(&data, &shape).unwrap();

    let dot = module.dot_product_optimize(&array_a, &array_b).unwrap();
    println!("{}", dot);

    // let data_a = array_a.get_heap();
    // let data_b = array_b.get_heap();
    // let dot = data_a
    //     .iter()
    //     .zip(data_b.iter())
    //     .map(|(&a, &b)| a * b)
    //     .sum::<f32>();
    // println!("{:?}", dot);
    //
    // println!("{:?}", &module.get_heap()[21..30]);

    // let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
    //     heap_size: HeapSize::Item(10000002),
    //     wgpu: arr_o_gpu::WgpuInit::ManualInit(arr_o_gpu::ManualInit {
    //         power: arr_o_gpu::Power::HighPerformance,
    //         memory: arr_o_gpu::Memory::Performance,
    //     }),
    // })
    // .unwrap();

    // let shape = [10000000];
    // let data = vec![2.; 10000000];
    // let array = module.array_from_vector(&data, &shape).unwrap();

    // let tick = std::time::SystemTime::now()
    //     .duration_since(UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    // let _dot = module.dot_product_optimize(&array, &array).unwrap();

    // let tock = std::time::SystemTime::now()
    //     .duration_since(UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    // println!("{}", tock - tick);
}
