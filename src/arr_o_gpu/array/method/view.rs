use crate::{ GpuArray, GpuArrayView };

impl GpuArray {
    pub fn view(&self) -> GpuArrayView<'_> {
        let shape = self.shape.clone();
        GpuArrayView {
            array: self,
            pointer: self.pointer,
            stride: self.stride.clone(),
            shape,
            offset: 0,
        }
    }
}
