use std::fmt::{ Debug, Display };

use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule, ArrayView, r };
use rotta_rs::{ Tensor, arrayy::matmul_nd, matmul };

fn main() {
    let module = ArrOgpuModule::default();

    let arr_a = ArangeArray::arange(0..100)
        .to_GpuArray_with_shape(&[2, 5, 10], &module)
        .unwrap();
    let arr_a = module.permute(&arr_a, &[0, 2, 1]).unwrap();
    println!("{}", arr_a.contiguous());

    let arr_b = ArangeArray::arange(0..10)
        .to_GpuArray_with_shape(&[10, 1], &module)
        .unwrap();
    let arr_b = module.broadcast_view(&arr_b, &[2, 10, 5]).unwrap();
    let arr_b = module.permute(&arr_b, &[0, 2, 1]).unwrap();
    let arr_b = arr_b.contiguous().get_heap();
    // println!("{}", arr_b);
    // let arr =
    // println!("{}", arr_b.contiguous());

    // let mul = module.matmul_nd_view(&arr_a, &arr_b).unwrap();
    // println!("{}", mul);

    // let mul_ = module.matmul_nd_view(&arr_a, &arr_b.contiguous()).unwrap();
    // println!("{}", mul_);

    // println!("{}", mul_a.get_heap() == mul_b.get_heap());

    //

    // let tensor_a = Tensor::from_vector(vec![2, 5, 10], arr_a.get_heap()).value();
    // let tensor_b = Tensor::from_vector(vec![2, 10, 5], arr_b.contiguous().get_heap()).value();
    // let tensor_mul = matmul_nd(&tensor_a, &tensor_b);

    // println!("{}", mul.get_heap() == tensor_mul.value)
}
