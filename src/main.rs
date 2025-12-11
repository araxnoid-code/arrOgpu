use std::fmt::{ Debug, Display };

use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r };
use rotta_rs::{ Tensor, arrayy::matmul_nd, matmul };

fn main() {
    let module = ArrOgpuModule::default();

    let arr_a = ArangeArray::arange(0..3 * 2 * 50 * 33)
        .to_GpuArray_with_shape(&[3, 2, 50, 33], &module)
        .unwrap();

    let arr_b = ArangeArray::arange(0..3 * 2 * 33 * 20)
        .to_GpuArray_with_shape(&[3, 2, 33, 20], &module)
        .unwrap();

    let mul = module.matmul_nd_view(&arr_a, &arr_b).unwrap();

    //

    // let tensor_a = Tensor::from_vector(vec![3, 2, 50, 33], arr_a.get_heap()).value();
    // let tensor_b = Tensor::from_vector(vec![3, 2, 33, 20], arr_b.get_heap()).value();
    // let tensor_mul = matmul_nd(&tensor_a, &tensor_b);

    // println!("{}", mul.get_heap() == tensor_mul.value)
}
