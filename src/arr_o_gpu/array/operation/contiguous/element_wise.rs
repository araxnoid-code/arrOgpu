use crate::{AbleElementWise, GpuArray};

impl GpuArray {
    pub fn add<A>(&self, arr: &A) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        A: AbleElementWise,
    {
        self.module().add(self, arr)
    }

    pub fn sub<A>(&self, arr: &A) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        A: AbleElementWise,
    {
        self.module().sub(self, arr)
    }

    pub fn mul<A>(&self, arr: &A) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        A: AbleElementWise,
    {
        self.module().mul(self, arr)
    }

    pub fn div<A>(&self, arr: &A) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        A: AbleElementWise,
    {
        self.module().div(self, arr)
    }
}
