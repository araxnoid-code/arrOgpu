use crate::{ArrayCompute, GpuArrayView};

impl<'a, A> GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    pub fn permute(
        &self,
        permute: &[u32],
    ) -> Result<GpuArrayView<'_, GpuArrayView<'_, A>>, crate::ArrOgpuErr> {
        self.module().permute(self, permute)
    }
}
