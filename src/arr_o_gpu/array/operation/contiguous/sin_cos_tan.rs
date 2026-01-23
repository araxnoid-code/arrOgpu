use crate::GpuArray;

impl GpuArray {
    pub fn sin(&self) -> Result<GpuArray, crate::ArrOgpuErr> {
        self.module().sin(self)
    }

    pub fn cos(&self) -> Result<GpuArray, crate::ArrOgpuErr> {
        self.module().cos(self)
    }

    pub fn tan(&self) -> Result<GpuArray, crate::ArrOgpuErr> {
        self.module().tan(self)
    }
}
