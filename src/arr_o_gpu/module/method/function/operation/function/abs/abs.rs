use crate::{ArrOgpuModule, ArrayCompute, CheckArrayType};

impl ArrOgpuModule {
    pub fn abs<'a, A>(&self, array: &'a A)
    where
        A: ArrayCompute + CheckArrayType<'a>,
    {
        match array.get() {
            crate::ArrayType::Contiguous(arr) => println!("ini contiguous"),
            crate::ArrayType::View(arr) => println!("ini view"),
        }
    }
}
