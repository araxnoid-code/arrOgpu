use std::fmt::{ Debug, Display };

use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule, ArrayView, r };

fn main() {
    let module = ArrOgpuModule::default();

    let arr_a = ArangeArray::arange(0..10)
        .to_GpuArray_with_shape(&[10], &module)
        .unwrap();
    println!("{}", arr_a);

    let arr_b = ArangeArray::arange(0..10)
        .to_GpuArray_with_shape(&[10], &module)
        .unwrap();
    println!("{}", arr_b);

    let product = module.dot_product_view(&arr_a, &arr_b).unwrap();
    println!("{}", product);
    let product_con = module.dot_product(&arr_a, &arr_b).unwrap();
    println!("{}", product_con)
}
