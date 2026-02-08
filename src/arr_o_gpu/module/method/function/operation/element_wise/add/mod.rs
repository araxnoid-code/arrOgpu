mod add_opt;
mod non_scalar;
mod scalar;

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, CheckArrayType, GpuArray,
    arr_o_gpu::module::method::function::operation::element_wise::skalar_operation::{
        AbleElementWise, ElementWiseOption,
    },
};

impl ArrOgpuModule {
    /// arguments can be arrays and floats can only be the second argument
    pub fn add<'a, A, B>(&self, array_a: &'a A, array_b: &'a B) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute + CheckArrayType<'a>,
        B: AbleElementWise,
    {
        match array_b.get_eble_element_wise() {
            ElementWiseOption::Array(array) => {
                return self.add_array(array_a, array);
            }
            ElementWiseOption::Skalar(meta_data_option) => {
                return self.add_skalar(array_a, meta_data_option);
            }
        }
    }
}
