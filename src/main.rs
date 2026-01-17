use std::vec;

fn main() {
    let shape = vec![2, 3, 4];
    let stride = vec![12, 4, 1];
    let mut axis = vec![0, 2];

    axis.sort();
    let mut out_shape = shape.clone();
    // let mut in_axis_shape = vec![1; shape.len()];
    // let mut sum_len = 1;
    for axis in axis.iter().rev() {
        // in_axis_shape[*axis] = shape[*axis];
        // sum_len *= shape[*axis];
        out_shape.remove(*axis);
    }
    // let out_stride = out_shape
    //     .iter()
    //     .enumerate()
    //     .map(|(i, _)| out_shape[i + 1..].iter().product::<u32>())
    //     .collect::<Vec<u32>>();
    // let in_axis_stride = in_axis_shape
    //     .iter()
    //     .enumerate()
    //     .map(|(i, _)| in_axis_shape[i + 1..].iter().product::<u32>())
    //     .collect::<Vec<u32>>();

    let len = out_shape.iter().product::<u32>();
    for x in 0..len {
        let first_axis = axis.first().unwrap();
        let last_axis = axis.last().unwrap();

        let counter = if *first_axis == 0 {
            stride[*last_axis]
        } else {
            stride[first_axis - 1] - stride[*last_axis]
        };

        // let counter = if axis.last().unwrap() == &0 {
        //     stride[0]
        // } else {
        //     stride[axis.last().unwrap() - &1] - stride[*axis.last().unwrap()]
        // };

        let x_index = x + counter * (x / stride[*axis.last().unwrap()]);

        println!("{}", counter);
    }
}
