use crate::GpuArray;

impl GpuArray {
    pub fn index(
        &self,
        index: &[u32],
    ) -> Result<crate::GpuArrayView<'_, GpuArray>, crate::ArrOgpuErr> {
        self.module.index(self, index)
    }
}
