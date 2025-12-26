use crate::GpuArray;

impl GpuArray {
    pub fn abs(&self) -> Result<GpuArray, crate::ArrOgpuErr> {
        self.module.abs(self)
    }
}
