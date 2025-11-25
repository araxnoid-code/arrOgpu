use crate::{ArrOgpuModule, FlatteTrait};

impl ArrOgpuModule {
    pub fn new<T>(&self, array: T)
    where
        T: FlatteTrait,
    {
    }
}
