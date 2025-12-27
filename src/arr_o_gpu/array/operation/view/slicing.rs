use crate::{
    ArrOgpuErr, GpuArray, SliceRange, SliceRangeNegativeAble, slice_negative_indexing_converter,
};

impl GpuArray {
    pub fn slicing(
        &self,
        slice: &[SliceRangeNegativeAble],
    ) -> Result<crate::GpuArrayView<'_, GpuArray>, crate::ArrOgpuErr> {
        let slice =
            slice_negative_indexing_converter(slice, &self.shape).map_err(ArrOgpuErr::from)?;

        self.module.slicing(self, &slice)
    }
}
