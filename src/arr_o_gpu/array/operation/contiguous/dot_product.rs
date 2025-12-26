use crate::{ArrayCompute, GpuArray};

impl GpuArray {
    pub fn dot<A>(&self, arr: &A) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        A: ArrayCompute,
    {
        self.module.dot_product(self, arr)
    }
}
