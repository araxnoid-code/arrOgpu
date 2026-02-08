use crate::ArrOgpuModule;

impl ArrOgpuModule {
    pub fn get_adapter_limit(&self) -> wgpu::Limits {
        self.wgpu_init().read().unwrap().get_adapter_limit()
    }

    pub fn get_device_limit(&self) -> wgpu::Limits {
        self.wgpu_init().read().unwrap().get_device_limit()
    }
}
