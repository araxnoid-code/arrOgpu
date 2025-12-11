use crate::{ GpuArray, GpuArrayView };

impl GpuArray {
    pub fn view(&self) -> GpuArrayView<'_, GpuArray> {
        let shape = self.shape.clone();

        let binding = self.module.array_data_binding(
            &self.pointer_to_arr(),
            &shape,
            &self.stride,
            &self.stride,
            &0
        );

        GpuArrayView {
            array: self,
            pointer: self.pointer,
            stride: self.stride.clone(),
            shape,
            offset: 0,
            binding,
        }
    }
}
