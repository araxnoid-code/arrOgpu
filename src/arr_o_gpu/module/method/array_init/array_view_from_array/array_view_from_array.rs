use crate::{ ArrOgpuModule, GpuArray, GpuArrayView };

impl ArrOgpuModule {
    pub fn array_view_from_array<'a>(&'a self, array: &'a GpuArray) -> GpuArrayView<'a> {
        GpuArrayView {
            array,
            pointer: array.pointer(),
            shape: array.shape.clone(),
            len: array.len() as u32,
        }
    }
}
