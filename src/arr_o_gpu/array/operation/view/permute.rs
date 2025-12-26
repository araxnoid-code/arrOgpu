use crate::GpuArray;

impl GpuArray {
    pub fn permute(
        &self,
        permute: &[u32],
    ) -> Result<crate::GpuArrayView<'_, GpuArray>, crate::ArrOgpuErr> {
        self.module.permute(self, permute)
    }
}
