use std::fmt::{ Debug, Display };

use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r };

fn main() {
    let module = ArrOgpuModule::default();

    let arr_a = ArangeArray::arange(0..25 * 50)
        .to_GpuArray_with_shape(&[25, 50], &module)
        .unwrap();
    // println!("{}", arr_a);

    let arr_b = ArangeArray::arange(0..500)
        .to_GpuArray_with_shape(&[50, 10], &module)
        .unwrap();
    // println!("{}", arr_a);

    let matmul_2d_old = module.matmul_2d(&arr_a, &arr_b).unwrap();
    println!("{}", matmul_2d_old);

    println!("=======");

    let matmul = module.matmul_2d_view(&arr_a, &arr_b).unwrap();
    println!("{}", matmul);

    println!("{}", matmul.get_heap() == matmul_2d_old.get_heap());
}
