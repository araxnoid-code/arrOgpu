use crate::{ArrayCompute, GpuArray, GpuArrayView};

pub enum ArrayType<'a> {
    Contiguous(&'a dyn ArrayCompute),
    View(&'a dyn ArrayCompute),
}

pub trait CheckArrayType<'a> {
    fn get(&'a self) -> ArrayType<'a>;
}

impl<'a> CheckArrayType<'a> for GpuArray {
    fn get(&'a self) -> ArrayType<'a> {
        ArrayType::Contiguous(self)
    }
}

impl<'a, A> CheckArrayType<'a> for GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    fn get(&'a self) -> ArrayType<'a> {
        ArrayType::View(self)
    }
}
