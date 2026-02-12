use crate::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};

#[test]
fn matmul_2d_error() {
    let module = ArrOgpuModule::default();

    let array_a = ArangeArray::arange(0..24)
        .to_GpuArray_with_shape(&[2, 3, 4], &module)
        .unwrap();

    let array_b = ArangeArray::arange(0..40)
        .to_GpuArray_with_shape(&[2, 5, 4], &module)
        .unwrap();

    let result = module.matmul(&array_a, &array_b);
    if let Ok(_) = result {
        panic!("error, valid input for matmul_nd should be [.., m, k] * [.., k, n]")
    }
    drop(array_a);
    drop(array_b);

    let array_a = ArangeArray::arange(0..32)
        .to_GpuArray_with_shape(&[2, 2, 2, 4], &module)
        .unwrap();

    let array_b = ArangeArray::arange(0..120)
        .to_GpuArray_with_shape(&[3, 2, 4, 5], &module)
        .unwrap();

    let result = module.matmul(&array_a, &array_b);
    if let Ok(_) = result {
        panic!(
            "error, valid input for matmul_nd should be [.., m, k] * [.., k, n] where the dimension before m and the dimension before k must be the same"
        )
    }
}
