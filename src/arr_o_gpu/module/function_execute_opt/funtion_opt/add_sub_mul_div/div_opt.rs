#[derive(Clone)]
pub struct DivOpt {
    pub compute_workgroup_size_x: u32,
}

impl Default for DivOpt {
    fn default() -> Self {
        Self {
            compute_workgroup_size_x: 256,
        }
    }
}
