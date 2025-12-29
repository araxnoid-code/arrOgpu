use crate::GpuArray;

impl GpuArray {
    pub fn index(
        &self,
        index: &[i32],
    ) -> Result<crate::GpuArrayView<'_, GpuArray>, crate::ArrOgpuErr> {
        self.module.index(self, &index)
    }
}
