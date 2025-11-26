use crate::{ArrOgpuModule, FlatteTrait};

impl ArrOgpuModule {
    pub fn new<T>(&self, array: T) -> Result<crate::GpuArray, crate::ArrOgpuErr>
    where
        T: FlatteTrait,
    {
        let (flatten, shape) = array.flatten();

        self.array_from_vector(&flatten, &shape)
    }
}
