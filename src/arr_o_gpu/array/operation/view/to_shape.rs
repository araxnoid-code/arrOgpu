use crate::GpuArray;

impl GpuArray {
    pub fn to_shape(
        &self,
        shape: &[u32],
    ) -> Result<crate::GpuArrayView<'_, GpuArray>, crate::ArrOgpuErr> {
        self.module().to_shape(self, shape)
    }
}
