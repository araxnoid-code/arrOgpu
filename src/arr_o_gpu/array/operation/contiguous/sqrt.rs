use crate::GpuArray;

impl GpuArray {
    pub fn sqrt(&self) -> Result<GpuArray, crate::ArrOgpuErr> {
        self.module.sqrt(self)
    }
}
