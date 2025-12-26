use crate::{ArrOgpuModule, GpuArray, GpuArrayView};

impl ArrOgpuModule {
    pub fn array_view_from_array<'a>(&self, array: &'a GpuArray) -> GpuArrayView<'a, GpuArray> {
        let shape = array.shape.clone();

        let binding = self.create_metadata_binding(
            &array.pointer_to_arr(),
            &shape,
            &array.stride,
            &array.stride,
            &0,
        );

        GpuArrayView {
            array,
            pointer: array.pointer(),
            stride: array.stride.clone(),
            shape,
            offset: 0,
            binding,
        }
    }
}
