use crate::{ArrayCompute, GpuArray, GpuArrayView};

impl<'a, A> GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    pub fn abs(&self) -> Result<GpuArray, crate::ArrOgpuErr> {
        self.module().abs(self)
    }
}
