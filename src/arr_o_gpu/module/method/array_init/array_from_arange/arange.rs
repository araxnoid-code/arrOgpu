use core::f32;
use std::{ ops::{ Range, RangeTo } };

use crate::{ ArrOgpuModule, GpuArray };

pub trait RangeArangeParams {
    fn range(self) -> Range<u32>;
}

impl RangeArangeParams for Range<u32> {
    fn range(self) -> Range<u32> {
        self
    }
}

impl RangeArangeParams for RangeTo<u32> {
    fn range(self) -> Range<u32> {
        0..self.end
    }
}

pub struct ArangeArray {
    range: Range<u32>,
}

impl ArangeArray {
    pub fn arange<T>(range: T) -> ArangeArray where T: RangeArangeParams + IntoIterator {
        let _range = range.range();

        Self { range: _range }
    }
}

#[allow(non_snake_case)]
pub trait ArangeIteratorTrait: Iterator {
    fn to_GpuArray(self, module: &ArrOgpuModule) -> Result<GpuArray, crate::ArrOgpuErr>;
    fn to_GpuArray_with_shape(
        self,
        shape: &[u32],
        module: &ArrOgpuModule
    ) -> Result<GpuArray, crate::ArrOgpuErr>;
}

impl<T: Iterator> ArangeIteratorTrait for T where T::Item: Into<f32> {
    fn to_GpuArray(self, module: &ArrOgpuModule) -> Result<GpuArray, crate::ArrOgpuErr> {
        let mut vector = vec![];
        for i in self {
            vector.push(i.into());
        }

        module.array_from_vector(&vector, &[vector.len() as u32])
    }

    fn to_GpuArray_with_shape(
        self,
        shape: &[u32],
        module: &ArrOgpuModule
    ) -> Result<GpuArray, crate::ArrOgpuErr> {
        let mut vector = vec![];
        for i in self {
            vector.push(i.into());
        }

        module.array_from_vector(&vector, shape)
    }
}

impl Iterator for ArangeArray {
    type Item = f32;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v) = self.range.next() { Some(v as f32) } else { None }
    }
}
