#[derive(Clone)]
pub struct AddOpt {
    pub compute_workgroup_size_x: u32,
}

impl Default for AddOpt {
    fn default() -> Self {
        Self {
            compute_workgroup_size_x: 256,
        }
    }
}
