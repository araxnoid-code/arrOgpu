use crate::GpuArray;

impl GpuArray {
    pub fn log2(&self) -> Result<GpuArray, crate::ArrOgpuErr> {
        self.module().log2(self)
    }
}
