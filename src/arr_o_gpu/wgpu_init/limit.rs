#[derive(Clone)]
pub struct WgpuLimits {
    pub max_storage_buffer_binding_size: u32,
    pub max_buffer_size: u64,
}

impl Default for WgpuLimits {
    fn default() -> Self {
        Self {
            max_storage_buffer_binding_size: 128 << 20, // 128MiB
            max_buffer_size: 256 << 20,                 // 256MiB
        }
    }
}
