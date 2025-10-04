// FLATTEN
// inlcuded vec<T> and [T; N]
pub trait FlatteTrait {
    fn flatten(&self) -> Vec<f32>;

    fn flatten_rec(&self, output: &mut Vec<f32>);
}

impl<T> FlatteTrait for Vec<T> where T: FlatteTrait {
    fn flatten(&self) -> Vec<f32> {
        let mut output: Vec<f32> = vec![];

        self.flatten_rec(&mut output);

        output
    }

    fn flatten_rec(&self, output: &mut Vec<f32>) {
        for item in self {
            item.flatten_rec(output);
        }
    }
}

impl FlatteTrait for f32 {
    fn flatten(&self) -> Vec<f32> {
        vec![]
    }

    fn flatten_rec(&self, output: &mut Vec<f32>) {
        output.push(*self);
    }
}

impl<T: FlatteTrait, const N: usize> FlatteTrait for [T; N] {
    fn flatten(&self) -> Vec<f32> {
        let mut output = vec![];
        self.flatten_rec(&mut output);
        output
    }

    fn flatten_rec(&self, output: &mut Vec<f32>) {
        for item in self {
            item.flatten_rec(output);
        }
    }
}

impl<T: FlatteTrait> FlatteTrait for &[T] {
    fn flatten(&self) -> Vec<f32> {
        let mut output = vec![];
        self.flatten_rec(&mut output);
        output
    }

    fn flatten_rec(&self, output: &mut Vec<f32>) {
        for item in *self {
            item.flatten_rec(output);
        }
    }
}
