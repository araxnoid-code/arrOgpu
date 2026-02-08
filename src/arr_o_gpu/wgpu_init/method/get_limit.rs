use crate::WgpuModule;

impl WgpuModule {
    pub fn get_adapter_limit(&self) -> wgpu::Limits {
        self.adapter.limits()
    }

    pub fn get_device_limit(&self) -> wgpu::Limits {
        self.device.limits()
    }
}
