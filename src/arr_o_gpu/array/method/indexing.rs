use crate::{ ArrayView, GpuArray, GpuArrayView };

impl GpuArray {
    pub fn index<A>(&self, index: &[u32]) -> Result<GpuArrayView<'_, GpuArray>, crate::ArrOgpuErr>
        where A: ArrayView
    {
        self.module.index::<GpuArray>(self, index)
    }
}
