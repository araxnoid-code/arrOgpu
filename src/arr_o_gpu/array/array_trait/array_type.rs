use crate::{ArrayCompute, GpuArray, GpuArrayView};

pub enum ArrayType<'a> {
    Contiguous(&'a dyn ArrayCompute),
    View(&'a dyn ArrayCompute),
}

pub trait CheckArrayType<'a> {
    fn check(&'a self) -> ArrayType<'a>;

    fn get(&'a self) -> &'a dyn ArrayCompute {
        match self.check() {
            ArrayType::Contiguous(arr) => arr,
            ArrayType::View(arr) => arr,
        }
    }
}

impl<'a> CheckArrayType<'a> for GpuArray {
    fn check(&'a self) -> ArrayType<'a> {
        ArrayType::Contiguous(self)
    }
}

impl<'a, A> CheckArrayType<'a> for GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    fn check(&'a self) -> ArrayType<'a> {
        ArrayType::View(self)
    }
}
