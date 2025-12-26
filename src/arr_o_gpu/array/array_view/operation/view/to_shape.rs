use crate::{ArrayCompute, GpuArrayView};

impl<'a, A> GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    pub fn to_shape(
        &self,
        shape: &[u32],
    ) -> Result<GpuArrayView<'_, GpuArrayView<'_, A>>, crate::ArrOgpuErr> {
        self.module().to_shape(self, shape)
    }
}
