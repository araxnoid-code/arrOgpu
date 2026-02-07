use std::sync::Arc;

use monagement::Allocated;

use crate::{ArrOgpuModule, MetadataCompound, SpaceType};

pub struct GpuArray {
    // module
    pub(crate) module: Arc<ArrOgpuModule>,

    // meta data
    pub(crate) length: usize,
    pub(crate) shape: Vec<u32>,
    pub(crate) stride: Vec<u32>,
    pub(crate) metadata_compound: Option<MetadataCompound>,

    // allocator update
    // pub(crate) pointer: (u32, u32),
    pub(crate) allocated: Allocated,
    // allocator update
}

impl GpuArray {
    pub fn module(&self) -> &Arc<ArrOgpuModule> {
        &self.module
    }

    pub fn pointer(&self) -> (u32, u32) {
        let range = self.allocated.get_range();
        (range.0 as u32, range.1 as u32)
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

    // pub fn space_type(&self) -> &SpaceType {
    //     &self.space_type
    // }

    pub fn dim(&self) -> usize {
        self.shape.len()
    }

    pub fn pointer_to_arr(&self) -> [u32; 2] {
        let pointer = self.pointer();
        [pointer.0 as u32, pointer.1 as u32]
    }

    pub fn metadata_compound(&self) -> Option<&MetadataCompound> {
        self.metadata_compound.as_ref()
    }
}
