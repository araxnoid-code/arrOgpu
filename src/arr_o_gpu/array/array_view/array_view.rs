use crate::GpuArray;

pub struct GpuArrayView<'a> {
    pub(crate) array: &'a GpuArray,
}
