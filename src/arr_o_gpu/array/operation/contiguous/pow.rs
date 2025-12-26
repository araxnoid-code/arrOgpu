use crate::{AblePowType, GpuArray};

impl GpuArray {
    pub fn pow<P>(&self, power: &P) -> Result<GpuArray, crate::ArrOgpuErr>
    where
        P: AblePowType,
    {
        self.module.pow(self, power)
    }
}
