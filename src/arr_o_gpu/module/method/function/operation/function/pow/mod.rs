use bytemuck::{Pod, Zeroable};

use crate::{ArrayCompute, GpuArray, GpuArrayView};

mod powf;
mod powi;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Power {
    pub(crate) counter: u32,
    pub(crate) scalar: f32,
    pub(crate) index: u32,
    pub(crate) padding: u32,
}

pub trait PowTrait {
    fn get(&self) -> Result<Power, &'static str>;
}

pub trait PowFloat {}
pub trait PowInt {}

impl PowTrait for f32 {
    fn get(&self) -> Result<Power, &'static str> {
        Ok(Power {
            counter: 0,
            index: 0,
            scalar: *self,
            padding: 0,
        })
    }
}

impl PowFloat for f32 {}

impl PowTrait for i32 {
    fn get(&self) -> Result<Power, &'static str> {
        Ok(Power {
            counter: 0,
            index: 0,
            scalar: *self as f32,
            padding: 0,
        })
    }
}
impl PowInt for i32 {}

impl PowTrait for GpuArray {
    fn get(&self) -> Result<Power, &'static str> {
        if self.dim() != 1 || self.shape()[0] != 1 {
            return Err("Power Must Be Scalar");
        }
        let index = self.offset() + self.pointer().0;
        Ok(Power {
            counter: 1,
            index,
            scalar: 0.,
            padding: 0,
        })
    }
}

impl PowFloat for GpuArray {}

impl<'a, A> PowTrait for GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    fn get(&self) -> Result<Power, &'static str> {
        if self.dim() != 1 || self.shape()[0] != 1 {
            return Err("Power Must Be Scalar");
        }
        let index = self.offset() + self.pointer().0;
        Ok(Power {
            counter: 1,
            index,
            scalar: 0.,
            padding: 0,
        })
    }
}
impl<'a, A> PowFloat for GpuArrayView<'a, A> where A: ArrayCompute {}
