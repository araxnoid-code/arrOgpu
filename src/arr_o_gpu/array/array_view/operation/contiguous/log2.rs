use crate::{ArrayCompute, GpuArray, GpuArrayView};

impl<'a, A> GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    pub fn log2(&self) -> Result<GpuArray, crate::ArrOgpuErr> {
        self.module().log2(self)
    }
}
