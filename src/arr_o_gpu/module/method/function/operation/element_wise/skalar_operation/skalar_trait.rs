use crate::ArrayView;

pub(crate) enum PointerOption {
    Skalar(f32),
    Pointer([u32; 2]),
}

pub(crate) enum ElementWiseOption<'a> {
    Skalar(PointerOption),
    Array(&'a dyn ArrayView),
}

pub trait AbleElementWise {
    fn get(&self) -> ElementWiseOption;
}

impl AbleElementWise for f32 {
    fn get(&self) -> ElementWiseOption {
        ElementWiseOption::Skalar(PointerOption::Skalar(*self))
    }
}

impl<A: ArrayView> AbleElementWise for A {
    fn get(&self) -> ElementWiseOption {
        let shape = self.shape();
        if shape.len() == 1 && shape[0] == 1 {
            ElementWiseOption::Skalar(PointerOption::Pointer(self.pointer_to_arr()))
        } else {
            ElementWiseOption::Array(self)
        }
    }
}
