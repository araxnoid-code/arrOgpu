use std::ops::Range;

fn main() {
    let data = (0..12).map(|v| v as f32).collect::<Vec<f32>>();
    let array = Array {
        data,
        shape: vec![2, 2, 3],
        stride: vec![6, 3, 1],
    };

    sum_axis(array, &[0]);
}

struct Array {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}

fn sum_axis(array: Array, axis: &[usize]) {
    let mut out_shape = vec![];
    let mut total_jump = 1;
    let mut iters = vec![];
    'a: for (dim, shape) in array.shape.iter().enumerate() {
        for axis in axis {
            if &dim == axis {
                total_jump *= shape;
                iters.push(1);
                continue 'a;
            }
        }

        iters.push(*shape);
        out_shape.push(*shape);
    }

    let iters = iters
        .iter()
        .enumerate()
        .map(|(i, _)| iters[i + 1..].iter().product())
        .collect::<Vec<usize>>();

    let stride = out_shape
        .iter()
        .enumerate()
        .map(|(i, _)| out_shape[i + 1..].iter().product())
        .collect::<Vec<usize>>();

    let total_unit = out_shape.iter().product::<usize>();
    for i in 0..total_unit {
        let mut slicing = vec![];
        let mut offset = 0;
        let mut indexing = 0;

        let mut index = 0;
        'a: for (ii, shape) in array.shape.iter().enumerate() {
            for axis in axis {
                if &ii == axis {
                    slicing.push(0..*shape);
                    continue 'a;
                }
            }

            let permute_start = (i / stride[index]) % shape;
            offset += permute_start * array.stride[ii];

            // for iii in 0..total_jump {
            //     let permute = (iii / iters[ii]) % 1;
            // }

            slicing.push(permute_start..permute_start + 1);
            index += 1;
        }

        println!("{:?}", slicing);
    }
}
