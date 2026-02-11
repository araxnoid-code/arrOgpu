#[derive(Clone)]
pub struct MatmulOpt {
    workgroup_size: u32,
}

impl Default for MatmulOpt {
    fn default() -> Self {
        Self { workgroup_size: 16 }
    }
}
