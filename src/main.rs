use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, WgpuLimits};

fn main() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        heap_size: arr_o_gpu::HeapSize::Item(33554445),
        limits: WgpuLimits {
            max_storage_buffer_binding_size: 256 << 20,
            ..Default::default()
        },
        wgpu: arr_o_gpu::WgpuInit::ManualInit(arr_o_gpu::ManualInit::default()),
    });
}
