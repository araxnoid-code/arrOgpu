use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, CheckArrayType, GpuArray,
    arr_o_gpu::module::method::function::operation::element_wise::skalar_operation::{
        AbleElementWise, ElementWiseOption,
    },
};

impl ArrOgpuModule {
    pub fn add<'a, A, B>(&self, array_a: &A, array_b: &B) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
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

    pub fn add_metadata<'a, A, B>(
        &self,
        array_a: &'a A,
        array_b: &'a B,
    ) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute + CheckArrayType<'a>,
        B: AbleElementWise,
    {
        match array_b.get_eble_element_wise() {
            ElementWiseOption::Array(array) => {
                return self.add_array_metadata(array_a, array);
            }
            ElementWiseOption::Skalar(meta_data_option) => {
                return self.add_skalar_metadata(array_a, meta_data_option);
            }
        }
    }
}
