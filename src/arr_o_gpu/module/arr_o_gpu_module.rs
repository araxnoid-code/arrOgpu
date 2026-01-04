use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use wgpu::{Buffer, ComputePipeline};

use crate::{Allocator, BindGroupCompound, arr_o_gpu::WgpuModule};

#[derive(Clone)]
pub struct ArrOgpuModule {
    pub(crate) allocator: Arc<RwLock<Allocator>>,
    pub(crate) maximum: Arc<u32>,
    pub(crate) wgpu_init: Arc<RwLock<WgpuModule>>,

    // Module Bind
    pub(crate) module_bind_group: Arc<BindGroupCompound>,

    // cache
    // // pipeline
    pub(crate) pipeline_cache: Arc<RwLock<HashMap<&'static str, ComputePipeline>>>,

    // Module Buffer
    // // heap
    pub(crate) heap_buffer: Arc<Buffer>,
    // // execute_array_cache
    pub(crate) execute_args: Arc<Buffer>,
    // // execute_array_cache
    pub(crate) static_cache: Arc<Buffer>,
}

// basic
impl ArrOgpuModule {
    pub fn allocator_read(&self) -> std::sync::RwLockReadGuard<'_, Allocator> {
        self.allocator.read().unwrap()
    }

    pub fn get_maximum(&self) -> u32 {
        *self.maximum
    }

    pub fn wgpu_init(&self) -> &Arc<RwLock<WgpuModule>> {
        &self.wgpu_init
    }

    pub fn heap_buffer(&self) -> &Buffer {
        &*self.heap_buffer
    }

    pub fn heap_binding(&self) -> &Arc<BindGroupCompound> {
        &self.module_bind_group
    }

    pub fn allocator_write(&self) -> std::sync::RwLockWriteGuard<'_, Allocator> {
        self.allocator.write().unwrap()
    }
}
