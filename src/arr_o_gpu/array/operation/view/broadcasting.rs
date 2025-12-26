use crate::GpuArray;

impl GpuArray {
    pub fn broadcast(
        &self,
        broadcast: &[u32],
    ) -> Result<crate::GpuArrayView<'_, GpuArray>, crate::ArrOgpuErr> {
        self.module.broadcast(self, broadcast)
    }
}
