#[derive(Clone)]
pub struct MulOpt {
    pub compute_workgroup_size_x: u32,
}

impl Default for MulOpt {
    fn default() -> Self {
        Self {
            compute_workgroup_size_x: 256,
        }
    }
}
