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

    // let matmul_2d_old = module.matmul_2d(&arr_a, &arr_b).unwrap();
    // println!("{}", matmul_2d_old);

    println!("=======");

    let matmul = module.matmul_2d_view(&arr_a, &arr_b).unwrap();
    println!("{}", matmul);

    let a = ndarray::Array2::from_shape_vec([25, 50], arr_a.get_heap()).unwrap();
    let b = ndarray::Array2::from_shape_vec([50, 10], arr_b.get_heap()).unwrap();
    let c = a.dot(&b);
    println!("{}", c);

    println!("{}", matmul.get_heap() == c.flatten().to_vec());
}
