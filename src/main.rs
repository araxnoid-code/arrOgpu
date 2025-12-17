use wgpu::wgc::id;

fn main() {
    let data = (0..12).map(|v| v as f32).collect::<Vec<f32>>();
    let array = Array {
        data,
        shape: vec![2, 2, 3],
        stride: vec![6, 3, 1],
    };

    sum_axis(array, &[0, 1]);
}

struct Array {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}

fn sum_axis(array: Array, axis: &[usize]) {
    let mut out_shape = vec![];
    let mut total_jump = 1;
    'a: for (dim, shape) in array.shape.iter().enumerate() {
        for axis in axis {
            if &dim == axis {
                total_jump *= shape;

                continue 'a;
            }
        }

        out_shape.push(*shape);
    }

    let out_iters = out_shape
        .iter()
        .enumerate()
        .map(|(i, _)| out_shape[i + 1..].iter().product::<usize>())
        .collect::<Vec<usize>>();

    // slicing
    let slice_shape = array.shape
        .iter()
        .enumerate()
        .map(|(i, s)| {
            for axis in axis {
                if axis == &i {
                    return *s;
                }
            }
            return 1;
        })
        .collect::<Vec<usize>>();

    let slicing_stride = slice_shape
        .iter()
        .enumerate()
        .map(|(i, _)| slice_shape[i + 1..].iter().product())
        .collect::<Vec<usize>>();

    for x in 0..out_shape.iter().product() {
        for y in 0..total_jump {
            let mut offset = 0;
            let mut index = 0;

            let mut idx = 0;
            for i in 0..array.shape.len() {
                let mut in_axis = false;
                for axis in axis {
                    if axis == &i {
                        in_axis = true;
                        break;
                    }
                }

                if in_axis {
                    let permute = (y / slicing_stride[i]) % array.shape[i];
                    println!("\n====> {:?}\n", y);
                    index += permute * array.stride[i];
                } else {
                    let permute = (x / out_iters[idx]) % out_shape[idx];

                    offset += permute * array.stride[i];
                    idx += 1;
                }
            }
            print!("{} ", offset + index);
        }
        println!("\n========");
    }
}
