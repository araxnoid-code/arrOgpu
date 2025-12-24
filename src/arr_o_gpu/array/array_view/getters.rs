use crate::{ArrayCompute, GpuArrayView};

impl<'a, A> GpuArrayView<'a, A>
where
    A: ArrayCompute,
{
    pub fn ref_array(&self) -> &A {
        self.array
    }

    pub fn pointer(&self) -> (u32, u32) {
        self.pointer
    }

    pub fn shape(&self) -> &Vec<u32> {
        &self.shape
    }

    pub fn offset(&self) -> u32 {
        self.offset
    }

    pub fn stride(&self) -> &Vec<u32> {
        &self.stride
    }
}
