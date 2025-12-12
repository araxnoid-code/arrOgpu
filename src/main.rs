use std::fmt::{ Debug, Display };

use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule, ArrayView, r };

fn main() {
    let module = ArrOgpuModule::default();

    let arr_a = ArangeArray::arange(0..200)
        .to_GpuArray_with_shape(&[40, 5], &module)
        .unwrap();
    let arr_a = arr_a.view();
    // let arr_a = module.slicing_view(&arr_a, &[r(100..150)]).unwrap();

    let arr_b = ArangeArray::arange(0..200)
        .to_GpuArray_with_shape(&[40, 5], &module)
        .unwrap();
    let arr_b = arr_b.view();
    // let arr_b = module.broadcast_view(&arr_b, &[2, 50]).unwrap();
    // let arr_b = module.index_view(&arr_b, &[1]).unwrap();

    let product = module.dot_product_view(&arr_a, &arr_b).unwrap();
    println!("{}", product);

    let product_con = module.dot_product(&arr_a.contiguous(), &arr_b.contiguous()).unwrap();
    println!("{}", product_con)
}
