use wgpu::Buffer;

use crate::{ arr_o_gpu::WgpuInit, Allocator, BindGroupCompound };

pub struct ArrOgpuModule {
    pub allocator: Allocator,
    pub maximum: u32,
    pub wgpu_init: WgpuInit,
    pub heap_buffer: Buffer,
    pub binding_compounds: Vec<BindGroupCompound>,
}
