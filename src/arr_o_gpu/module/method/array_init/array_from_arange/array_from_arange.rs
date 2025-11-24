use std::{ops::Range, sync::Arc};

use crate::{ArangeArray, ArrOgpuModule, RangeArangeParams};

impl ArrOgpuModule {
    pub fn arange<T>(&self, range: T) -> ArangeArray
    where
        T: RangeArangeParams + IntoIterator,
    {
        ArangeArray::arange(range)
    }
}
