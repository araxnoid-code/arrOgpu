use bytemuck::{Pod, Zeroable};
use wgpu::{BindGroup, BindGroupLayout, Buffer};

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
}

pub struct MetadataCompound {
    pub(crate) buffer: Buffer,
    pub(crate) bind_group_layout: BindGroupLayout,
    pub(crate) bind_group: BindGroup,
}
