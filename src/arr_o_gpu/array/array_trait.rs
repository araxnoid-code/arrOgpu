// FLATTEN
// inlcuded vec<T> and [T; N]
pub trait FlatteTrait {
    fn flatten(&self) -> Vec<f32>;

    fn flatten_rec(&self, output: &mut Vec<f32>, shape: &mut Vec<u32>, dim: usize);
}

impl FlatteTrait for f32 {
    fn flatten(&self) -> Vec<f32> {
        vec![]
    }

    fn flatten_rec(&self, output: &mut Vec<f32>, shape: &mut Vec<u32>, dim: usize) {
        output.push(*self);
    }
}

impl<T> FlatteTrait for Vec<T>
where
    T: FlatteTrait,
{
    fn flatten(&self) -> Vec<f32> {
        let mut output: Vec<f32> = vec![];
        let mut shape: Vec<u32> = vec![];
        let dim = 0;

        self.flatten_rec(&mut output, &mut shape, dim);

        shape.reverse();
        println!("{:?}", shape);

        output
    }

    fn flatten_rec(&self, output: &mut Vec<f32>, shape: &mut Vec<u32>, dim: usize) {
        if let Some(i) = shape.get_mut(dim) {
            *i += 1;
        } else {
            shape.push(1);
        }

        for item in self {
            item.flatten_rec(output, shape, dim + 1);
        }
    }
}

// impl<T> FlatteTrait for &Vec<T>
// where
//     T: FlatteTrait,
// {
//     fn flatten(&self) -> Vec<f32> {
//         let mut output: Vec<f32> = vec![];

//         self.flatten_rec(&mut output);

//         output
//     }

//     fn flatten_rec(&self, output: &mut Vec<f32>) {
//         for item in *self {
//             item.flatten_rec(output);
//         }
//     }
// }

// impl<T: FlatteTrait, const N: usize> FlatteTrait for [T; N] {
//     fn flatten(&self) -> Vec<f32> {
//         let mut output = vec![];
//         self.flatten_rec(&mut output);
//         output
//     }

//     fn flatten_rec(&self, output: &mut Vec<f32>) {
//         for item in self {
//             item.flatten_rec(output);
//         }
//     }
// }

// impl<T: FlatteTrait, const N: usize> FlatteTrait for &[T; N] {
//     fn flatten(&self) -> Vec<f32> {
//         let mut output = vec![];
//         self.flatten_rec(&mut output);
//         output
//     }

//     fn flatten_rec(&self, output: &mut Vec<f32>) {
//         for item in *self {
//             item.flatten_rec(output);
//         }
//     }
// }

// impl<T: FlatteTrait> FlatteTrait for &[T] {
//     fn flatten(&self) -> Vec<f32> {
//         let mut output = vec![];
//         let mut shape: Vec<u32> = vec![];
//         self.flatten_rec(&mut output);
//         output
//     }

//     fn flatten_rec(&self, output: &mut Vec<f32>) {
//         for item in *self {
//             item.flatten_rec(output);
//         }
//     }
// }
