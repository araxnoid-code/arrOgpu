use crate::{AbleElementWise, ArrayCompute, GpuArray, GpuArrayView};

impl<'a, A> GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    pub fn add<B>(&self, arr: &B) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        B: AbleElementWise,
    {
        self.module().add(self, arr)
    }

    pub fn sub<B>(&self, arr: &B) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        B: AbleElementWise,
    {
        self.module().sub(self, arr)
    }

    pub fn mul<B>(&self, arr: &B) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        B: AbleElementWise,
    {
        self.module().mul(self, arr)
    }

    pub fn div<B>(&self, arr: &B) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        B: AbleElementWise,
    {
        self.module().div(self, arr)
    }
}
