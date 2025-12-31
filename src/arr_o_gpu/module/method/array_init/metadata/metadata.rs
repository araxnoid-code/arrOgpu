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
    pub(crate) shape: [u32; 12],         // 48
    pub(crate) stride: [u32; 12],        // 48
    pub(crate) origin_stride: [u32; 12], // 48
    //
    //
    pub(crate) padding_3: [u32; 20],
}

pub struct MetadataCompound {
    pub(crate) buffer: Buffer,
}

// struct ArrayMetadata{
//     // --- //
// 	pointer: vec2<u32>, // 8
// 	len: u32, // 4
// 	offset: u32, // 4
// 	// --- //

// 	// --- //
// 	dim: u32, // 4
// 	padding0: u32, // 4
// 	padding1: u32, // 4
// 	padding2: u32, // 4
// 	// --- //

// 	shape: array<vec4<u32>, 3>, // 48
// 	stride: array<vec4<u32>, 3>, // 48
// 	origin_stride: array<vec4<u32>, 3>, // 48

// 	// padding
// 	padding: array<vec4<u32>, 5>,
// }
