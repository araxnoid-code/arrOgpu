use std::sync::{ Arc, RwLock };

use wgpu::Buffer;

use crate::{ arr_o_gpu::WgpuInit, Allocator, BindGroupCompound };

#[derive(Clone)]
pub struct ArrOgpuModule {
    pub(crate) allocator: Arc<RwLock<Allocator>>,
    pub(crate) maximum: Arc<u32>,
    pub(crate) wgpu_init: Arc<RwLock<WgpuInit>>,
    pub(crate) heap_buffer: Arc<Buffer>,
    pub(crate) heap_binding: Arc<BindGroupCompound>,
    // pub(crate) binding_compounds: Arc<RwLock<Vec<BindGroupCompound>>>,
}

// basic
impl ArrOgpuModule {
    pub fn allocator_read(&self) -> std::sync::RwLockReadGuard<'_, Allocator> {
        self.allocator.read().unwrap()
    }

    pub fn get_maximum(&self) -> u32 {
        *self.maximum
    }

    pub fn wgpu_init(&self) -> &Arc<RwLock<WgpuInit>> {
        &self.wgpu_init
    }

    pub fn heap_buffer(&self) -> &Buffer {
        &*self.heap_buffer
    }

    pub fn heap_binding(&self) -> &Arc<BindGroupCompound> {
        &self.heap_binding
    }

    pub fn allocator_write(&self) -> std::sync::RwLockWriteGuard<'_, Allocator> {
        self.allocator.write().unwrap()
    }
}
