use crate::GpuArray;

pub struct GpuArrayView<'a> {
    pub(crate) array: &'a GpuArray,
    pub(crate) pointer: (u32, u32),
    pub(crate) shape: Vec<u32>,
    pub(crate) offset: u32,
    pub(crate) offset_list: Vec<u32>,
}
