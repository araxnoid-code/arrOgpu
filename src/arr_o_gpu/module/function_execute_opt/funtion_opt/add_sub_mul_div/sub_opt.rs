#[derive(Clone)]
pub struct SubOpt {
    pub compute_workgroup_size_x: u32,
}

impl Default for SubOpt {
    fn default() -> Self {
        Self {
            compute_workgroup_size_x: 256,
        }
    }
}
