use std::vec;

fn main() {
    let gpu_virtualizer = GpuVirtualizer {};

    let shape = vec![2, 3, 4];
    let stride = vec![12, 4, 1];
    let axis = vec![1, 2];

    let mut out_shape = shape.clone();
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
        .map(|(i, _)| out_shape[i + 1..].iter().product::<u32>())
        .collect::<Vec<u32>>();
    let in_axis_stride = in_axis_shape
        .iter()
        .enumerate()
        .map(|(i, _)| in_axis_shape[i + 1..].iter().product::<u32>())
        .collect::<Vec<u32>>();

    println!("{in_axis_shape:?}");
    println!("{in_axis_stride:?}");

    gpu_virtualizer.running(16, 16, 1, |x, y, _| {
        let total_x = out_shape.iter().product::<u32>();
        if x < total_x && y < sum_len {
            let mut is_axis = false;
            let mut idx = 0;
            print!("( ");
            for (d, s) in shape.iter().enumerate() {
                for d_axis in &axis {
                    if *d_axis == d {
                        is_axis = true;
                        break;
                    }
                }

                if is_axis {
                    let permute = (y / in_axis_stride[d]) % in_axis_shape[d];
                    print!(" {} ", permute);
                } else {
                    let permute = (x / out_stride[idx]) % out_shape[idx];
                    idx += 1;
                    print!(" {} ", permute)
                }
                is_axis = false;
            }
            print!(" )")
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
