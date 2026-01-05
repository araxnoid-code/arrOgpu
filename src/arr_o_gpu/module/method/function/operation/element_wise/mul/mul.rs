use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, CheckArrayType, GpuArray,
    arr_o_gpu::module::method::function::operation::element_wise::skalar_operation::{
        AbleElementWise, ElementWiseOption,
    },
};

impl ArrOgpuModule {
    /// element-wise subtraction operations, input in the form of arrays and floats can be used as input in the second argument
    pub fn mul<'a, A, B>(&self, array_a: &'a A, array_b: &'a B) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute + CheckArrayType<'a>,
        B: AbleElementWise,
    {
        match array_b.get_eble_element_wise() {
            ElementWiseOption::Array(array) => {
                return self.mul_array(array_a, array);
            }
            ElementWiseOption::Skalar(meta_data_option) => {
                return self.mul_skalar(array_a, meta_data_option);
            }
        }
    }
}
