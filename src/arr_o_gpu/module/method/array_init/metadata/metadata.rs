use bytemuck::{Pod, Zeroable};
use wgpu::Buffer;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ArrayMetadata {
    pub(crate) pointer: [u32; 2],
    pub(crate) len: u32,
    pub(crate) offset: u32,
    pub(crate) dim: u32,
    pub(crate) shape: [u32; 10],
    pub(crate) stride: [u32; 10],
    pub(crate) origin_stride: [u32; 10],
    pub(crate) padding: [u32; 29],
}

pub struct MetadataCompound {
    pub(crate) buffer: Buffer,
}
