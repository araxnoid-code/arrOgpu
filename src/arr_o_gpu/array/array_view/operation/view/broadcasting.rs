use crate::{ArrayCompute, GpuArrayView};

impl<'a, A> GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    pub fn broadcast(
        &self,
        broadcast: &[u32],
    ) -> Result<GpuArrayView<'_, GpuArrayView<'_, A>>, crate::ArrOgpuErr> {
        self.module().broadcast(self, broadcast)
    }
}
