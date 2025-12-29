use arr_o_gpu::{ArrOgpuModule, HeapSize};

fn main() {

    // let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
    //     heap_size: HeapSize::Item(100000),
    //     wgpu: arr_o_gpu::WgpuInit::ManualInit(arr_o_gpu::ManualInit {
    //         power: arr_o_gpu::Power::HighPerformance,
    //         memory: arr_o_gpu::Memory::Performance,
    //     }),
    // })
    // .unwrap();

    // let shape = [10];
    // let data = vec![1.; 10];
    // let array = module.array_from_vector(&data, &shape).unwrap();
    // drop(data);

    // let tick = std::time::SystemTime::now()
    //     .duration_since(UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    // let _dot = module.dot_product_optimize(&array, &array).unwrap();

    // let tock = std::time::SystemTime::now()
    //     .duration_since(UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    // println!("{}ms", tock - tick);
}
