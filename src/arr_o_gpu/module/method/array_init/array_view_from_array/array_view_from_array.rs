use crate::{ArrOgpuModule, GpuArray, GpuArrayView};

impl ArrOgpuModule {
    pub fn array_view_from_array<'a>(
        &self,
        array: &'a GpuArray,
    ) -> Result<GpuArrayView<'a, GpuArray>, crate::ArrOgpuErr> {
        array.view()
    }
}
