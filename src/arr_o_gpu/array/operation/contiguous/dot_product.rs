use crate::{ArrayCompute, CheckArrayType, GpuArray};

impl GpuArray {
    pub fn dot<'a, A>(&'a self, arr: &'a A) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        A: ArrayCompute + CheckArrayType<'a>,
    {
        self.module().dot_product(self, arr)
    }
}
