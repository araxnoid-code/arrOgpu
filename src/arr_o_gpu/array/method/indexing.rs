use crate::GpuArray;

impl GpuArray {
    pub fn index(&self, index: &[u32]) -> Result<GpuArray, crate::ArrOgpuErr> {
        self.module.index(self, index)
    }
}
