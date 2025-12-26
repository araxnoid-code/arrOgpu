use crate::{AblePowType, ArrayCompute, GpuArray, GpuArrayView};

impl<'a, A> GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    pub fn pow<P>(&self, power: &P) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        P: AblePowType,
    {
        self.module().pow(self, power)
    }
}
