use crate::{GpuArray, SliceRange};

impl GpuArray {
    pub fn slicing(
        &self,
        slice: &[SliceRange],
    ) -> Result<crate::GpuArrayView<'_, GpuArray>, crate::ArrOgpuErr> {
        self.module.slicing(self, slice)
    }
}
