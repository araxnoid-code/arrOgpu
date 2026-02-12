#[derive(Clone)]
pub struct MatmulOpt {
    pub workgroup_size: u32,
}

impl Default for MatmulOpt {
    fn default() -> Self {
        Self { workgroup_size: 16 }
    }
}
