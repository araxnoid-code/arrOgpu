use crate::GpuArray;

impl GpuArray {
    pub fn sum(&self) -> Result<GpuArray, crate::ArrOgpuErr> {
        self.module.sum(self)
    }

    pub fn sum_axis(&self, axis: &[u32]) -> Result<GpuArray, crate::ArrOgpuErr> {
        self.module.sum_axis(self, axis)
    }

    pub fn sum_axis_keep_dim(&self, axis: &[u32]) -> Result<GpuArray, crate::ArrOgpuErr> {
        self.module.sum_axis_keep_dim(self, axis)
    }
}
