use crate::{ ArrOgpuModule, GpuArray, GpuArrayView };

impl ArrOgpuModule {
    pub fn array_view_from_array<'a>(&'a self, array: &'a GpuArray) -> GpuArrayView<'_, GpuArray> {
        let shape = array.shape.clone();
        GpuArrayView {
            array,
            pointer: array.pointer(),
            stride: array.stride.clone(),
            shape,
            offset: 0,
        }
    }
}
