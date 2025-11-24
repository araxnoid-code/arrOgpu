use core::f32;
use std::{
    cell::RefCell,
    ops::{Index, Range, RangeTo},
    rc::Rc,
    sync::Arc,
    usize,
};

use crate::{ArrOgpuModule, GpuArray};

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
    range: (Range<u32>, ArangeShape),
}

impl ArangeArray {
    pub fn arange<T>(range: T) -> ArangeArray
    where
        T: RangeArangeParams + IntoIterator,
    {
        // let _range = range.range();
        let _range = range.range();
        let len = vec![_range.end - _range.start];
        let shape = ArangeShape {
            shape: Rc::new(RefCell::new(len)),
        };

        Self {
            range: (_range, shape),
        }
    }
}

#[allow(non_snake_case)]
pub trait ArangeIteratorTrait: Iterator {
    fn to_GpuArray(self, module: &ArrOgpuModule) -> Result<GpuArray, crate::ArrOgpuErr>;
    fn shape(self, shape: &[u32]) -> Self;
}

impl<T: Iterator> ArangeIteratorTrait for T
where
    T::Item: ArangeIteratorItem,
{
    fn to_GpuArray(self, module: &ArrOgpuModule) -> Result<GpuArray, crate::ArrOgpuErr> {
        let mut vector = vec![];
        let mut shape = vec![];
        for i in self {
            vector.push(i.item());
            if shape.is_empty() {
                let data = i.shape().borrow().clone();
                shape = data;
            }
        }

        println!("{:?}", vector);
        module.array_from_vector(&vector, &shape)
    }

    fn shape(mut self, shape: &[u32]) -> Self {
        let mut peekable = self.by_ref().peekable();
        peekable.peek();

        self
    }
}

pub trait ArangeIteratorItem {
    fn item(&self) -> f32;

    fn shape(&self) -> Rc<RefCell<Vec<u32>>>;

    fn update_shape(&mut self, new_shape: &[u32]);
}

impl ArangeIteratorItem for (f32, ArangeShape) {
    fn item(&self) -> f32 {
        self.0
    }

    fn shape(&self) -> Rc<RefCell<Vec<u32>>> {
        self.1.shape.clone()
    }

    fn update_shape(&mut self, new_shape: &[u32]) {
        // *self.1.shape.borrow_mut() = new_shape.to_vec();
    }
}

impl Iterator for ArangeArray {
    type Item = (f32, ArangeShape);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(v) = self.range.0.next() {
            Some((v as f32, self.range.1.clone()))
        } else {
            None
        }
    }
}

#[derive(Clone)]
pub struct ArangeShape {
    pub(crate) shape: Rc<RefCell<Vec<u32>>>,
}

// impl ArangeShape {
// pub(crate) fn update_shape(&mut self, new_shape: &[u32]) {
//     *self.shape.borrow_mut() = new_shape.to_vec();
// }
// }

// impl Index<usize> for ArangeShape {
//     type Output = u32;
//     fn index(&self, index: usize) -> &Self::Output {
//         &self.shape[index]
//     }
// }
