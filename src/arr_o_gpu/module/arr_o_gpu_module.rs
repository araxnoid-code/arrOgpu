use crate::arr_o_gpu::WgpuInit;

pub struct ArrOgpuModule {
    // allocator:
    wgpu_init: WgpuInit,
}

impl ArrOgpuModule {
    pub fn init() -> Self {
        Self {
            wgpu_init: WgpuInit::init(),
        }
    }
}
