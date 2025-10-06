use std::sync::{ Arc, RwLock };

use crate::ArrOgpuModule;

pub struct GpuArray {
    pub(crate) module: Arc<ArrOgpuModule>,
    pub(crate) pointer: (usize, usize),
    pub(crate) length: usize,
    pub(crate) shape: Vec<u32>,
}

impl GpuArray {
    pub fn module(&self) -> &ArrOgpuModule {
        &*self.module
    }

    pub fn pointer(&self) -> (usize, usize) {
        self.pointer
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn shape(&self) -> &Vec<u32> {
        &self.shape
    }
}
