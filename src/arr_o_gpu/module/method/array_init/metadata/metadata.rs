use bytemuck::{Pod, Zeroable};
use wgpu::Buffer;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ArrayMetadata {
    //
    pub(crate) pointer: [u32; 2], // 8
    pub(crate) len: u32,          // 4
    pub(crate) offset: u32,       // 4
    //
    //
    pub(crate) dim: u32,       // 4
    pub(crate) padding_0: u32, // 4
    pub(crate) padding_1: u32, // 4
    pub(crate) padding_2: u32, // 4
    //
    //
    pub(crate) shape: [u32; 8],    // 32
    pub(crate) stride: [u32; 8],   // 32
    pub(crate) o_stride: [u32; 8], // 32
    //
    pub(crate) m_n_shape: [u32; 8],
    pub(crate) m_n_o_stride: [u32; 8],
    //
    pub(crate) padding_3: [u32; 16],
}

pub struct MetadataCompound {
    pub(crate) buffer: Buffer,
}
