use bytemuck::{Pod, Zeroable};

use crate::{ArrayCompute, GpuArray, GpuArrayView};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PowerF {
    pub(crate) counter: u32,
    pub(crate) scalar: f32,
    pub(crate) index: u32,
    pub(crate) padding: u32,
}

pub trait PowFloat {
    fn get(&self) -> Result<PowerF, &'static str>;
}

impl PowFloat for f32 {
    fn get(&self) -> Result<PowerF, &'static str> {
        Ok(PowerF {
            counter: 0,
            index: 0,
            scalar: *self,
            padding: 0,
        })
    }
}

impl PowFloat for GpuArray {
    fn get(&self) -> Result<PowerF, &'static str> {
        if self.dim() != 1 || self.shape()[0] != 1 {
            return Err("Power Must Be Scalar");
        }
        let index = self.offset() + self.pointer().0;
        Ok(PowerF {
            counter: 1,
            index,
            scalar: 0.,
            padding: 0,
        })
    }
}

impl<'a, A> PowFloat for GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    fn get(&self) -> Result<PowerF, &'static str> {
        if self.dim() != 1 || self.shape()[0] != 1 {
            return Err("Power Must Be Scalar");
        }
        let index = self.offset() + self.pointer().0;
        Ok(PowerF {
            counter: 1,
            index,
            scalar: 0.,
            padding: 0,
        })
    }
}
