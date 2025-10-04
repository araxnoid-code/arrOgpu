use std::sync::{ Arc, RwLock };

use wgpu::Buffer;

use crate::{ arr_o_gpu::WgpuInit, Allocator, BindGroupCompound };

pub struct ArrOgpuModule {
    pub allocator: Arc<RwLock<Allocator>>,
    pub maximum: u32,
    pub wgpu_init: WgpuInit,
    pub heap_buffer: Arc<Buffer>,
    pub binding_compounds: Arc<RwLock<Vec<BindGroupCompound>>>,
}

// basic
impl ArrOgpuModule {
    pub fn allocator(&self) -> std::sync::RwLockWriteGuard<'_, Allocator> {
        self.allocator.write().unwrap()
    }
}
