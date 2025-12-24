use crate::{ArrayCompute, GpuArray, GpuArrayView};

impl GpuArray {
    pub fn index<A>(&self, index: &[u32]) -> Result<GpuArrayView<'_, GpuArray>, crate::ArrOgpuErr>
    where
        A: ArrayCompute,
    {
        self.module.index::<GpuArray>(self, index)
    }
}
