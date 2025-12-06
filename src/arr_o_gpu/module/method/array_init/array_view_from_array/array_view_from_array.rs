use crate::{ ArrOgpuModule, GpuArray, GpuArrayView };

impl ArrOgpuModule {
    pub fn array_view_from_array<'a>(&'a self, array: &'a GpuArray) -> GpuArrayView<'a> {
        let shape = array.shape.clone();
        GpuArrayView {
            array,
            pointer: array.pointer(),
            offset_list: vec![0; shape.len()],
            shape,
            offset: 0,
        }
    }
}
