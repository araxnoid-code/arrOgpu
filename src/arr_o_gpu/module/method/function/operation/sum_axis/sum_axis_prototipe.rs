fn main() {
    let shape = [[2, 3, 4, 0], [0, 0, 0, 0]];
    let stride = [[12, 4, 1, 0], [0, 0, 0, 0]];
    let dim = 3;

    let axis = [[0, 0, 0, 0], [0, 0, 0, 0]];

    let out_shape = [[3, 4, 0, 0], [0, 0, 0, 0]];
    let out_stride = [[4, 1, 0, 0], [0, 0, 0, 0]];
    let out_dim = 2;
    let len = 12;

    for x in 0..32 {
        for y in 0..32 {
            if x < len {
                let sum_len = get_sum_len(&axis, dim, &shape);
                if y < sum_len {
                    let in_axis_shape = get_in_axis_shape(&shape, dim, &axis);
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

fn get_sum_len(axis_list: &[[u32; 4]; 2], dim: u32, shape: &[[u32; 4]; 2]) -> u32 {
    let mut len = 1;
    for i in 0..8 {
        if i >= dim as usize {
            break;
        }

        let axis = axis_list[i >> 2][i & 3];
        let zero = (axis != 0 || i == 0) as u32;
        len *= 1 * (1 - zero) + shape[(axis >> 2) as usize][(axis & 3) as usize] * zero;
    }
    len
}

fn get_in_axis_shape(shape: &[[u32; 4]; 2], dim: u32, axis_list: &[[u32; 4]; 2]) -> [u32; 9] {
    let mut in_axis_shape = [1, 1, 1, 1, 1, 1, 1, 1, 0];

    for i in 0..8 {
        if i >= dim as usize {
            break;
        }

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
        if i >= dim as usize {
            break;
        }

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
