use crate::{ArrayCompute, GpuArray};

impl GpuArray {
    pub fn matmul<A>(&self, arr: &A) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        A: ArrayCompute,
    {
        self.module.matmul_nd(self, arr)
    }
}
