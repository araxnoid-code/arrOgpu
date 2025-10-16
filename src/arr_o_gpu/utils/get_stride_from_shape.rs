pub(crate) fn get_stride_from_shape(shape: &[u32]) -> Vec<u32> {
    let mut stride = vec![];
    for i in 0..shape.len() {
        stride.push(shape[i + 1..].into_iter().product::<u32>());
    }
    stride
}
