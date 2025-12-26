use crate::{ArrayCompute, GpuArrayView};

impl<'a, A> GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    pub fn index(
        &self,
        index: &[u32],
    ) -> Result<GpuArrayView<'_, GpuArrayView<'_, A>>, crate::ArrOgpuErr> {
        self.module().index(self, index)
    }
}
