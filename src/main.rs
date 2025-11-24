use std::ops::{Range, RangeFull};

use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, SlicingRangeTrait};

fn main() {
    let module = ArrOgpuModule::default();
    let array = module
        .arange(10..20)
        .map(|v| v)
        .step_by(1)
        .shape(&[5, 2])
        .to_GpuArray(&module)
        .unwrap();

    // .to_GpuArray(&module)
    // .unwrap();

    // println!("{}", array)

    // let data = myStruct::init().into_iter().map(|v| v);
}

// struct myStruct<'a> {
//     data: (Vec<i32>, &'a [usize]),
//     index: usize,
// }

// impl<'a> myStruct<'a> {
//     pub fn init() -> Self {
//         Self {
//             data: (vec![1, 2, 3, 4, 5], &[1, 5]),
//             index: 1,
//         }
//     }
// }

// impl<'a> Iterator for myStruct<'a> {
//     type Item = (i32, &'a [usize]);
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.index < self.data.0.len() {
//             let data = Some((self.data.0[self.index], self.data.1));
//             self.index += 1;
//             data
//         } else {
//             None
//         }
//     }
// }
