use std::sync::{ Arc, RwLock };

use crate::ArrOgpuModule;

pub struct GpuArray<'a> {
    pub module: Arc<RwLock<ArrOgpuModule>>,
    pub pointer: (usize, usize),
    pub length: usize,
    pub shape: &'a [u32],
}
