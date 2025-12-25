use crate::{
    ArrayCompute, arr_o_gpu::module::method::function::operation::element_wise::MetaDataOption,
};

pub trait AblePowType {
    fn get(&self) -> MetaDataOption<'_>;
}

impl AblePowType for f32 {
    fn get(&self) -> MetaDataOption<'_> {
        MetaDataOption::Skalar(*self)
    }
}

impl<A> AblePowType for A
where
    A: ArrayCompute,
{
    fn get(&self) -> MetaDataOption<'_> {
        MetaDataOption::Array(self)
    }
}
