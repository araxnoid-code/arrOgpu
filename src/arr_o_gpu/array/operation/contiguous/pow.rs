use crate::{GpuArray, PowFloat, PowInt, PowTrait};

impl GpuArray {
    pub fn powf<P>(&self, power: &P) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        P: PowTrait + PowFloat,
    {
        self.module().powf(self, power)
    }

    pub fn powi<P>(&self, power: &P) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        P: PowTrait + PowInt,
    {
        self.module().powi(self, power)
    }
}
