// use arr_o_gpu::*;

fn main() {
    // let _module = ArrOgpuModule::init();

    let arr: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![6.0, 7.0, 8.0, 9.0, 0.0]];
    println!("{:?}", arr.flatten());
}

pub trait FlatteTrait {
    fn flatten(&self) -> Vec<f32>;

    fn flatten_rec(&self, output: &mut Vec<f32>);
}

impl<T> FlatteTrait for Vec<T>
where
    T: FlatteTrait,
{
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
