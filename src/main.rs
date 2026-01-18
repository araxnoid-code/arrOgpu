fn main() {
    let shape = [[2, 3, 4, 0], [0, 0, 0, 0]];
    let stride = [[12, 4, 1, 0], [0, 0, 0, 0]];
    let dim = 3;

    let axis = [[0, 1, 2, 0], [0, 0, 0, 0]];

    let out_shape = [[1, 0, 0, 0], [0, 0, 0, 0]];
    let out_stride = [[1, 0, 0, 0], [0, 0, 0, 0]];
    let out_dim = 1;
    let len = 1;

    for x in 0..256 {
        for y in 0..256 {
            if x < len {
                let sum_len = get_sum_len(&axis, &shape);
                if y < sum_len {
                    let in_axis_shape = get_in_axis_shape(&shape, &axis);
                    let in_axis_stride = get_in_axis_stride(&in_axis_shape, dim);
                    let mark_stride = mark_stride(&shape, &stride, dim, &axis);

                    let mut index = 0;
                    for i in 0..8 {
                        let axis = axis[i >> 2][i & 3];
                        if axis != 0 || i == 0 {
                            let permute =
                                (y / in_axis_stride[axis as usize]) % in_axis_shape[axis as usize];
                            index += permute * stride[(axis >> 2) as usize][(axis & 3) as usize];
                        }
                    }

                    for i in 0..out_dim {
                        let permute = (x / out_stride[i >> 2][i & 3]) % out_shape[i >> 2][i & 3];
                        index += permute * mark_stride[i];
                    }

                    print!(" {}, ", index);
                }
            }
        }
        println!()
    }
}

fn get_sum_len(axis_list: &[[u32; 4]; 2], shape: &[[u32; 4]; 2]) -> u32 {
    let mut len = 1;
    for i in 0..8 {
        let axis = axis_list[i >> 2][i & 3];
        let zero = (axis != 0 || i == 0) as u32;
        len *= 1 * (1 - zero) + shape[(axis >> 2) as usize][(axis & 3) as usize] * zero;
    }
    len
}

fn get_in_axis_shape(shape: &[[u32; 4]; 2], axis_list: &[[u32; 4]; 2]) -> [u32; 9] {
    let mut in_axis_shape = [1, 1, 1, 1, 1, 1, 1, 1, 0];

    for i in 0..8 {
        let axis = axis_list[i >> 2][i & 3];
        let index = select(8, axis, axis != 0 || i == 0);
        in_axis_shape[index as usize] = shape[(axis >> 2) as usize][(axis & 3) as usize];
    }

    in_axis_shape
}

fn get_in_axis_stride(in_axis_shape: &[u32; 9], dim: u32) -> [u32; 8] {
    let mut in_axis_stride = [1, 1, 1, 1, 1, 1, 1, 1];

    for i in 0..dim {
        for ii in i + 1..dim {
            in_axis_stride[i as usize] *= in_axis_shape[ii as usize];
        }
    }

    in_axis_stride
}

fn mark_stride(
    shape: &[[u32; 4]; 2],
    stride: &[[u32; 4]; 2],
    dim: u32,
    axis_list: &[[u32; 4]; 2],
) -> [u32; 8] {
    let mut mark_stride = [0, 0, 0, 0, 0, 0, 0, 0];

    let mut mark_shape = shape.clone();
    for i in 0..8 {
        let axis = axis_list[i >> 2][i & 3];
        if axis != 0 || i == 0 {
            mark_shape[(axis >> 2) as usize][(axis & 3) as usize] = 0;
        }
    }

    let mut skip = 0;
    for i in 0..dim {
        if mark_shape[(i >> 2) as usize][(i & 3) as usize] != 0 {
            let stride = stride[(i >> 2) as usize][(i & 3) as usize];
            mark_stride[(i - skip) as usize] = stride;
        } else {
            skip += 1;
        }
    }

    mark_stride
}

fn select<T>(_false: T, _true: T, condition: bool) -> T {
    if condition { _true } else { _false }
}

// /////////////////////////////

// use std::vec;

// fn main() {
//     let gpu_virtualizer = GpuVirtualizer {};

//     let shape = vec![2, 3, 4];

//     let stride = vec![12, 4, 1];

//     let axis = vec![1, 2];

//     let mut out_shape = shape
//         .iter()
//         .enumerate()
//         .map(|(i, s)| (*s, stride[i]))
//         .collect::<Vec<(u32, u32)>>();

//     let mut in_axis_shape = vec![1; shape.len()];

//     let mut sum_len = 1;

//     for axis in axis.iter().rev() {
//         in_axis_shape[*axis] = shape[*axis];

//         sum_len *= shape[*axis];

//         out_shape.remove(*axis);
//     }

//     let out_stride = out_shape
//         .iter()
//         .enumerate()
//         .map(|(i, _)| {
//             out_shape[i + 1..]
//                 .iter()
//                 .map(|(shape, _)| *shape)
//                 .product::<u32>()
//         })
//         .collect::<Vec<u32>>();

//     let in_axis_stride = in_axis_shape
//         .iter()
//         .enumerate()
//         .map(|(i, _)| in_axis_shape[i + 1..].iter().product::<u32>())
//         .collect::<Vec<u32>>();

//     gpu_virtualizer.running(16, 16, 1, |x, y, _| {
//         let total_x = out_shape.iter().map(|(shape, _)| *shape).product::<u32>();

//         if x < total_x && y < sum_len {
//             let mut index = 0;

//             for d_axis in &axis {
//                 let permute = (y / in_axis_stride[*d_axis]) % in_axis_shape[*d_axis];

//                 index += permute * stride[*d_axis];
//             }

//             for (d, (out_shape, stride)) in out_shape.iter().enumerate() {
//                 let permute = (x / out_stride[d]) % out_shape;

//                 index += permute * stride;
//             }

//             print!("{} ", index);
//         }
//     });
// }

// struct GpuVirtualizer {}

// impl GpuVirtualizer {
//     pub fn running(&self, x: u32, y: u32, z: u32, f: impl Fn(u32, u32, u32)) {
//         for x in 0..x {
//             for y in 0..y {
//                 for z in 0..z {
//                     f(x, y, z);
//                 }
//             }

//             println!()
//         }
//     }
// }
