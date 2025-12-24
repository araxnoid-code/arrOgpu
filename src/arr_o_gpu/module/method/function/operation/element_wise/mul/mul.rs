use crate::arr_o_gpu::module::method::function::operation::element_wise::skalar_operation::*;

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, GpuArray,
    arr_o_gpu::module::method::function::operation::element_wise::skalar_operation::AbleElementWise,
};

impl ArrOgpuModule {
    pub fn mul<'a, A, B>(&self, array_a: &A, array_b: &B) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
        B: AbleElementWise,
    {
        match array_b.get() {
            ElementWiseOption::Array(array) => {
                return self.mul_array(array_a, array);
            }
            ElementWiseOption::Skalar(meta_data_option) => {
                return self.mul_skalar(array_a, meta_data_option);
            }
        }
    }
}
