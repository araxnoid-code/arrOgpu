use std::ops::{Range, RangeFull};

use arr_o_gpu::{ArangeArray, ArrOgpuModule, SlicingRangeTrait};

fn main() {
    // let array = ArangeArr::ini(1..10).collect::<Arr>();
    // let data = ArangeArr;
    let data = ArangeArray::arange(0..10)
        .map(|v| v + 10.)
        .collect::<Vec<f32>>();

    println!("{:?}", data);
}

// struct ArangeArr {
//     pub vector: Vec<f32>,
//     pub index: usize,
// }

// impl ArangeArr {
//     pub fn ini(range: Range<usize>) -> ArangeArr {
//         Self {
//             vector: range.map(|v| v as f32).collect(),
//             index: 0,
//         }
//     }
// }

// impl Iterator for ArangeArr {
//     type Item = f32;
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.index < self.vector.len() {
//             let item = Some(self.vector[self.index]);
//             self.index += 1;
//             return item;
//         } else {
//             return None;
//         }
//     }
// }

// struct Arr {
//     vector: Vec<f32>,
// }

// trait FloatArr {
//     fn get_float(&self) -> f32;
// }

// impl FloatArr for f32 {
//     fn get_float(&self) -> f32 {
//         *self as f32
//     }
// }

// impl<A: FloatArr> FromIterator<A> for Arr {
//     fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
//         let mut data = Vec::new();

//         for i in iter {
//             data.push(i.get_float());
//         }

//         Self { vector: data }
//     }
// }
