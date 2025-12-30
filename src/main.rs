use std::time::UNIX_EPOCH;

use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, HeapSize};

fn main() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        heap_size: HeapSize::Item(100000),
        wgpu: arr_o_gpu::WgpuInit::ManualInit(arr_o_gpu::ManualInit {
            power: arr_o_gpu::Power::LowPower,
            memory: arr_o_gpu::Memory::MemoryUsage,
        }),
    })
    .unwrap();

    let array_a = ArangeArray::arange(0..24).to_GpuArray(&module).unwrap();
    let array_b = ArangeArray::arange(24..48).to_GpuArray(&module).unwrap();

    let array_a_view = array_a.slicing(&[]);
}
