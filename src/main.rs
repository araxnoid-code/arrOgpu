use std::fmt::{ Debug, Display };

use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r };

fn main() {
    let module = ArrOgpuModule::default();

    let arr_a = ArangeArray::arange(0..15)
        .to_GpuArray_with_shape(&[5, 3], &module)
        .unwrap();
    println!("{}", arr_a);

    let slicing_a = module.slicing_view(&arr_a, &[r(1..3)]).unwrap();
    let slicing_a = module.permute(&slicing_a, &[1, 0]).unwrap();
    let slicing_b = module.slicing_view(&arr_a, &[r(3..5)]).unwrap();
    let slicing_b = module.permute(&slicing_b, &[1, 0]).unwrap();

    let sub = module.mul_view(&slicing_a, &slicing_b).unwrap();
    println!("{}", sub);

    println!("{}", slicing_a.contiguous());
    println!("{}", slicing_b.contiguous());

    // let arr_b = ArangeArray::arange(0..15)
    //     .to_GpuArray_with_shape(&[5, 3], &module)
    //     .unwrap();
    // println!("{}", arr_b);

    // let arr = module.add_view(&arr_a, &arr_b).unwrap();
    // println!("{}", arr);

    // println!("{:?}", module.get_heap());
}
