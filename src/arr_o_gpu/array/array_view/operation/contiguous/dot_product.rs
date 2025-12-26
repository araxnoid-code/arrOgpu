use crate::{ArrayCompute, GpuArray, GpuArrayView};

impl<'a, A> GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    pub fn dot<B>(&self, arr: &B) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        B: ArrayCompute,
    {
        self.module().dot_product(self, arr)
    }
}
