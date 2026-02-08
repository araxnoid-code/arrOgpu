#[derive(Clone)]
pub struct WgpuLimits {
    pub max_storage_buffer_binding_size: u32,
    pub max_buffer_size: u64,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_workgroups_per_dimension: u32,
}

impl Default for WgpuLimits {
    fn default() -> Self {
        Self {
            max_storage_buffer_binding_size: 128 << 20, // 128MiB
            max_buffer_size: 256 << 20,                 // 256MiB
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_workgroups_per_dimension: 65535,
        }
    }
}
