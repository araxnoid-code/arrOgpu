use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, CheckArrayType, GpuArray,
    arr_o_gpu::module::method::function::operation::function::pow::AblePowType,
};

impl ArrOgpuModule {
    pub fn pow<'a, A, P>(&self, array: &'a A, pow: &P) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute + CheckArrayType<'a>,
        P: AblePowType,
    {
        match array.check() {
            crate::ArrayType::Contiguous(arr) => self.pow_contiguous(arr, pow),
            crate::ArrayType::View(arr) => self.pow_view(arr, pow),
        }
    }
}
