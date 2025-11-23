use std::{
    ops::{Range, RangeTo},
    sync::Arc,
};

use crate::{ArrOgpuModule, GpuArray};

pub trait RangeArangeParams {
    fn range(self) -> Range<usize>;
}

impl RangeArangeParams for Range<usize> {
    fn range(self) -> Range<usize> {
        self
    }
}

impl RangeArangeParams for RangeTo<usize> {
    fn range(self) -> Range<usize> {
        0..self.end
    }
}

pub struct ArangeArray {
    vector: Vec<f32>,
    shape: Option<Vec<u32>>,
    index: usize,
    module: Arc<ArrOgpuModule>,
}

impl ArangeArray {
    pub fn arange<T>(range: T, module: Arc<ArrOgpuModule>) -> ArangeArray
    where
        T: RangeArangeParams + IntoIterator,
        T::Item: Into<usize>,
    {
        let vector = range
            .into_iter()
            .map(|v| v.into() as f32)
            .collect::<Vec<f32>>();

        Self {
            vector,
            shape: None,
            index: 0,
            module,
        }
    }

    pub fn shape(&mut self, shape: &[u32]) {
        self.shape = Some(shape.to_vec());
    }

    #[allow(non_snake_case)]
    pub fn to_GpuArray(self) -> Result<GpuArray, crate::ArrOgpuErr> {
        let shape = self.shape.clone();
        let module = self.module.clone();
        let vector = self.collect::<Vec<f32>>();
        let shape = shape.unwrap_or(vec![vector.len() as u32]);
        module.array_from_vector(&vector, &shape)
    }
}

pub trait ArrangeIteratorTrait: Iterator {
    fn to_GpuArray(self) -> Result<GpuArray, crate::ArrOgpuErr>;
}

// impl<T: Iterator> ArrangeIteratorTrait for T {
//     fn to_GpuArray(self) -> Result<GpuArray, crate::ArrOgpuErr> {}
// }

impl Iterator for ArangeArray {
    type Item = f32;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.vector.len() {
            let item = Some(self.vector[self.index]);
            self.index += 1;
            item
        } else {
            None
        }
    }
}
