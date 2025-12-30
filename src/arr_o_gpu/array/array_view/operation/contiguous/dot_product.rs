use crate::{ArrayCompute, CheckArrayType, GpuArray, GpuArrayView};

impl<'a, A> GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    pub fn dot<'b, B>(&'b self, arr: &'b B) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        B: ArrayCompute + CheckArrayType<'b>,
    {
        self.module().dot_product(self, arr)
    }
}
