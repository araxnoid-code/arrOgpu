#[derive(Clone)]
pub struct MatmulOpt {
    pub workgroup_size_x_and_y: u32,
}

impl Default for MatmulOpt {
    fn default() -> Self {
        Self {
            workgroup_size_x_and_y: 16,
        }
    }
}
