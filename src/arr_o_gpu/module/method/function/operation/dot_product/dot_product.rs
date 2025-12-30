use crate::{ArrOgpuErr, ArrOgpuModule, ArrayCompute, ArrayType, CheckArrayType, GpuArray};

impl ArrOgpuModule {
    pub fn dot_product<'a, A, B>(
        &self,
        array_a: &'a A,
        array_b: &'a B,
    ) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute + CheckArrayType<'a>,
        B: ArrayCompute + CheckArrayType<'a>,
    {
        match (array_a.check(), array_b.check()) {
            (ArrayType::Contiguous(arr_a), ArrayType::Contiguous(arr_b)) => {
                self.dot_product_contiguous(arr_a, arr_b)
            }

            _ => self.dot_product_view(array_a, array_b),
        }
    }
}
