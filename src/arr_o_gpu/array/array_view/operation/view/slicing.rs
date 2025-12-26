use crate::{ArrayCompute, GpuArrayView, SliceRange};

impl<'a, A> GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    pub fn slicing(
        &self,
        slice: &[SliceRange],
    ) -> Result<GpuArrayView<'_, GpuArrayView<'_, A>>, crate::ArrOgpuErr> {
        self.module().slicing(self, slice)
    }
}
