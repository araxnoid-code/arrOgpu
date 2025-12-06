use crate::{ GpuArray, GpuArrayView };

impl GpuArray {
    pub fn view(&self) -> GpuArrayView<'_> {
        let shape = self.shape.clone();
        GpuArrayView {
            array: self,
            pointer: self.pointer,
            offset_list: vec![0; shape.len()],
            shape,
            offset: 0,
        }
    }
}
