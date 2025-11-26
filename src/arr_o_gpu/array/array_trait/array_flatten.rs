use std::usize;

// FLATTEN
// inlcuded vec<T> and [T; N]
pub trait FlatteTrait {
    fn flatten(&self) -> (Vec<f32>, Vec<u32>);

    fn flatten_rec(&self, output: &mut Vec<f32>, shape: &mut Vec<u32>, dim: usize);
}

impl FlatteTrait for f32 {
    fn flatten(&self) -> (Vec<f32>, Vec<u32>) {
        (vec![], vec![])
    }

    fn flatten_rec(&self, output: &mut Vec<f32>, _: &mut Vec<u32>, _: usize) {
        output.push(*self);
    }
}

impl<T> FlatteTrait for Vec<T>
where
    T: FlatteTrait,
{
    fn flatten(&self) -> (Vec<f32>, Vec<u32>) {
        let mut output: Vec<f32> = vec![];
        let mut shape: Vec<u32> = vec![];
        let dim = 0;

        self.flatten_rec(&mut output, &mut shape, dim);

        shape.reverse();
        (output, shape)
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

impl<T> FlatteTrait for &Vec<T>
where
    T: FlatteTrait,
{
    fn flatten(&self) -> (Vec<f32>, Vec<u32>) {
        let mut output: Vec<f32> = vec![];
        let mut shape: Vec<u32> = vec![];
        let dim = 0;

        self.flatten_rec(&mut output, &mut shape, dim);

        shape.reverse();
        (output, shape)
    }

    fn flatten_rec(&self, output: &mut Vec<f32>, shape: &mut Vec<u32>, dim: usize) {
        if let Some(i) = shape.get_mut(dim) {
            *i += 1;
        } else {
            shape.push(1);
        }

        for item in *self {
            item.flatten_rec(output, shape, dim + 1);
        }
    }
}

impl<T, const N: usize> FlatteTrait for [T; N]
where
    T: FlatteTrait,
{
    fn flatten(&self) -> (Vec<f32>, Vec<u32>) {
        let mut output: Vec<f32> = vec![];
        let mut shape: Vec<u32> = vec![];
        let dim = 0;

        self.flatten_rec(&mut output, &mut shape, dim);

        shape.reverse();
        (output, shape)
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

impl<T, const N: usize> FlatteTrait for &[T; N]
where
    T: FlatteTrait,
{
    fn flatten(&self) -> (Vec<f32>, Vec<u32>) {
        let mut output: Vec<f32> = vec![];
        let mut shape: Vec<u32> = vec![];
        let dim = 0;

        self.flatten_rec(&mut output, &mut shape, dim);

        shape.reverse();
        (output, shape)
    }

    fn flatten_rec(&self, output: &mut Vec<f32>, shape: &mut Vec<u32>, dim: usize) {
        if let Some(i) = shape.get_mut(dim) {
            *i += 1;
        } else {
            shape.push(1);
        }

        for item in *self {
            item.flatten_rec(output, shape, dim + 1);
        }
    }
}

impl<T> FlatteTrait for [T]
where
    T: FlatteTrait,
{
    fn flatten(&self) -> (Vec<f32>, Vec<u32>) {
        let mut output: Vec<f32> = vec![];
        let mut shape: Vec<u32> = vec![];
        let dim = 0;

        self.flatten_rec(&mut output, &mut shape, dim);

        shape.reverse();
        (output, shape)
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

impl<T> FlatteTrait for &[T]
where
    T: FlatteTrait,
{
    fn flatten(&self) -> (Vec<f32>, Vec<u32>) {
        let mut output: Vec<f32> = vec![];
        let mut shape: Vec<u32> = vec![];
        let dim = 0;

        self.flatten_rec(&mut output, &mut shape, dim);

        shape.reverse();
        (output, shape)
    }

    fn flatten_rec(&self, output: &mut Vec<f32>, shape: &mut Vec<u32>, dim: usize) {
        if let Some(i) = shape.get_mut(dim) {
            *i += 1;
        } else {
            shape.push(1);
        }

        for item in *self {
            item.flatten_rec(output, shape, dim + 1);
        }
    }
}
