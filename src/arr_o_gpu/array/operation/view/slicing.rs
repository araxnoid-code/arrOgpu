use crate::{ArrOgpuErr, GpuArray, SliceRangeNegativeAble, slice_negative_indexing_converter};

impl GpuArray {
    pub fn slicing(
        &self,
        slice: &[SliceRangeNegativeAble],
    ) -> Result<crate::GpuArrayView<'_, GpuArray>, crate::ArrOgpuErr> {
        self.module().slicing(self, &slice)
    }
}
