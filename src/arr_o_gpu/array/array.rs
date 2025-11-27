use std::sync::Arc;

use crate::{ArrOgpuModule, SpaceType};

pub struct GpuArray {
    pub(crate) module: Arc<ArrOgpuModule>,
    pub(crate) pointer: (usize, usize),
    pub(crate) length: usize,
    pub(crate) shape: Vec<u32>,
    pub(crate) stride: Vec<u32>,

    //
    pub(crate) space_type: SpaceType,
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

    pub fn stride(&self) -> &Vec<u32> {
        &self.stride
    }

    pub fn space_type(&self) -> &SpaceType {
        &self.space_type
    }

    pub fn dim(&self) -> usize {
        self.shape.len()
    }

    pub fn pointer_to_arr(&self) -> [u32; 2] {
        [self.pointer.0 as u32, self.pointer.1 as u32]
    }
}
