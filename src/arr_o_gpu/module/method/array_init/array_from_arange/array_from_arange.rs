use std::sync::Arc;

use crate::{ArangeArray, ArrOgpuModule, RangeArangeParams};

impl ArrOgpuModule {
    pub fn arange<T>(&self, range: T) -> ArangeArray
    where
        T: RangeArangeParams + IntoIterator,
        T::Item: Into<usize>,
    {
        ArangeArray::arange(range, Arc::new(self.clone()))
    }
}
