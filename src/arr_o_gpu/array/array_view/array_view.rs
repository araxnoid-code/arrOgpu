use crate::{ ArrayView, GpuArray };

pub struct GpuArrayView<'a, A> where A: ArrayView {
    pub(crate) array: &'a A,
    pub(crate) pointer: (u32, u32),
    pub(crate) shape: Vec<u32>,
    pub(crate) stride: Vec<u32>,
    pub(crate) offset: u32,
}
