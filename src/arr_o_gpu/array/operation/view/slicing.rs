use crate::{GpuArray, SliceRangeNegativeAble};

impl GpuArray {
    pub fn slicing(
        &self,
        slice: &[SliceRangeNegativeAble],
    ) -> Result<crate::GpuArrayView<'_, GpuArray>, crate::ArrOgpuErr> {
        self.module().slicing(self, &slice)
    }
}
