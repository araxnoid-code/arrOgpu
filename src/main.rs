use std::vec;

fn main() {
    let gpu_virtualizer = GpuVirtualizer {};

    let shape = vec![2, 3, 4];
    let stride = vec![12, 4, 1];
    let axis = vec![1, 2];

    let mut out_shape = shape
        .iter()
        .enumerate()
        .map(|(i, s)| (*s, stride[i]))
        .collect::<Vec<(u32, u32)>>();
    let mut in_axis_shape = vec![1; shape.len()];
    let mut sum_len = 1;
    for axis in axis.iter().rev() {
        in_axis_shape[*axis] = shape[*axis];
        sum_len *= shape[*axis];
        out_shape.remove(*axis);
    }
    let out_stride = out_shape
        .iter()
        .enumerate()
        .map(|(i, _)| {
            out_shape[i + 1..]
                .iter()
                .map(|(shape, _)| *shape)
                .product::<u32>()
        })
        .collect::<Vec<u32>>();
    let in_axis_stride = in_axis_shape
        .iter()
        .enumerate()
        .map(|(i, _)| in_axis_shape[i + 1..].iter().product::<u32>())
        .collect::<Vec<u32>>();

    gpu_virtualizer.running(16, 16, 1, |x, y, _| {
        let total_x = out_shape.iter().map(|(shape, _)| *shape).product::<u32>();
        if x < total_x && y < sum_len {
            let mut index = 0;
            for d_axis in &axis {
                let permute = (y / in_axis_stride[*d_axis]) % in_axis_shape[*d_axis];
                index += permute * stride[*d_axis];
            }

            for (d, (out_shape, stride)) in out_shape.iter().enumerate() {
                let permute = (x / out_stride[d]) % out_shape;
                index += permute * stride;
            }

            print!("{} ", index);
        }
    });
}

struct GpuVirtualizer {}

impl GpuVirtualizer {
    pub fn running(&self, x: u32, y: u32, z: u32, f: impl Fn(u32, u32, u32)) {
        for x in 0..x {
            for y in 0..y {
                for z in 0..z {
                    f(x, y, z);
                }
            }
            println!()
        }
    }
}
