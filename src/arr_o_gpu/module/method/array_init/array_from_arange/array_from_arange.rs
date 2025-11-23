use std::ops::{Range, RangeTo};

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
}

impl ArangeArray {
    pub fn arange<T>(range: T) -> ArangeArray
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
        }
    }

    pub fn shape(&mut self, shape: &[u32]) {
        self.shape = Some(shape.to_vec());
    }
}

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
