use std::time::UNIX_EPOCH;

use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, HeapSize};

fn main() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        heap_size: HeapSize::Item(5000000),
        wgpu: arr_o_gpu::WgpuInit::ManualInit(arr_o_gpu::ManualInit {
            power: arr_o_gpu::Power::HighPerformance,
            memory: arr_o_gpu::Memory::Performance,
        }),
    })
    .unwrap();

    let shape = vec![4194304];
    let data = vec![2.; 4194304];
    let array = module.array_from_vector(&data, &shape).unwrap();

    let tick = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let dot = module.dot_product_optimize(&array, &array).unwrap();
    let tock = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();

    println!("{}", tock - tick);
}
// 16777216
// 16777216.0
