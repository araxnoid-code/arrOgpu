use crate::{ArrayCompute, CheckArrayType};

pub enum MetaDataOption<'a> {
    Skalar(f32),
    Array(&'a dyn ArrayCompute),
}

pub enum ElementWiseOption<'a> {
    Skalar(MetaDataOption<'a>),
    Array(&'a dyn ArrayCompute),
}

pub trait AbleElementWise {
    fn get_eble_element_wise(&self) -> ElementWiseOption<'_>;
}

impl AbleElementWise for f32 {
    fn get_eble_element_wise(&self) -> ElementWiseOption<'_> {
        ElementWiseOption::Skalar(MetaDataOption::Skalar(*self))
    }
}

impl<'a, A: ArrayCompute> AbleElementWise for A {
    fn get_eble_element_wise(&self) -> ElementWiseOption<'_> {
        let shape = self.shape();
        if shape.len() == 1 && shape[0] == 1 {
            ElementWiseOption::Skalar(MetaDataOption::Array(self))
        } else {
            ElementWiseOption::Array(self)
        }
    }
}
