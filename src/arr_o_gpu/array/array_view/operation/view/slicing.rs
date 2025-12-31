use crate::{
    ArrOgpuErr, ArrayCompute, GpuArrayView, SliceRange, SliceRangeNegativeAble,
    slice_negative_indexing_converter,
};

impl<'a, A> GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    pub fn slicing(
        &self,
        slice: &[SliceRangeNegativeAble],
    ) -> Result<GpuArrayView<'_, GpuArrayView<'_, A>>, crate::ArrOgpuErr> {
        let slice = slice_negative_indexing_converter(slice, &self.shape)
            .map_err(|err| ArrOgpuErr::Slicing(err))?;
        self.module().slicing(self, &slice)
    }
}
