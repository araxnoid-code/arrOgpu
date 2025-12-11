use std::fmt::{ Debug, Display };

use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r };

fn main() {
    let module = ArrOgpuModule::default();

    let arr_a = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[2, 2, 3], &module)
        .unwrap();
    println!("{}", arr_a);

    let arr_b = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[2, 3, 2], &module)
        .unwrap();
    println!("{}", arr_b);
    // let arr_b = module.permute(&arr_b, &[1, 0]).unwrap();
    // let arr_b = module.broadcast_view(&arr_b, &[3, 3, 10]).unwrap();
    // let arr_b = module.index_view(&arr_b, &[1]).unwrap();
    // let arr_b = module.slicing_view(&arr_b, &[r(..), r(3..6)]).unwrap();

    let mul = module.matmul_nd_view(&arr_a, &arr_b).unwrap();
    print!("{}", mul);

    // println!("==============");
    // let mul = module.matmul_2d_view(&arr_a, &arr_b.contiguous()).unwrap();
    // print!("{}", mul);

    // println!("\n{:?}", module.get_heap());
}
