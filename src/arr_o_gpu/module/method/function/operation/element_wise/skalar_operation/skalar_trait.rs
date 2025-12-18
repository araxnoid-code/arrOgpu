use crate::ArrayView;

pub enum MetaDataOption<'a> {
    Skalar(f32),
    Array(&'a dyn ArrayView),
}

pub enum ElementWiseOption<'a> {
    Skalar(MetaDataOption<'a>),
    Array(&'a dyn ArrayView),
}

pub trait AbleElementWise {
    fn get(&self) -> ElementWiseOption<'_>;
}

impl AbleElementWise for f32 {
    fn get(&self) -> ElementWiseOption<'_> {
        ElementWiseOption::Skalar(MetaDataOption::Skalar(*self))
    }
}

impl<A: ArrayView> AbleElementWise for A {
    fn get(&self) -> ElementWiseOption<'_> {
        let shape = self.shape();
        if shape.len() == 1 && shape[0] == 1 {
            ElementWiseOption::Skalar(MetaDataOption::Array(self))
        } else {
            ElementWiseOption::Array(self)
        }
    }
}
