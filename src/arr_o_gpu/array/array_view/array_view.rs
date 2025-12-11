use wgpu::{ BindGroup, BindGroupLayout };

use crate::ArrayView;

pub struct GpuArrayView<'a, A> where A: ArrayView {
    // meta data
    pub(crate) array: &'a A,
    pub(crate) pointer: (u32, u32),
    pub(crate) shape: Vec<u32>,
    pub(crate) stride: Vec<u32>,
    pub(crate) offset: u32,

    // binding
    pub(crate) binding: (BindGroupLayout, BindGroup),
}
