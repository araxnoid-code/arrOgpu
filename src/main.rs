use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r };

fn main() {
    let module = ArrOgpuModule::default();

    let arr = ArangeArray::arange(0..30)
        .to_GpuArray_with_shape(&[2, 5, 3], &module)
        .unwrap();
    println!("{}", arr);

    let arr = module.index_view(&arr, &[1]).unwrap();
    let arr = module.permute(&arr, &[1, 0]).unwrap();
    let arr = module.slicing_view(&arr, &[r(1..3), r(2..)]).unwrap();
    let arr = module.broadcast_view(&arr, &[3, 2, 3]).unwrap();
    let arr = module.slicing_view(&arr, &[r(1..3), r(1..2)]).unwrap();
    let arr = module.permute(&arr, &[2, 1, 0]).unwrap();
    let arr = module.broadcast_view(&arr, &[3, 3]).unwrap().contiguous();
    println!("{}", arr);
    let arr = module.reshape(&arr, &[3, 6]).unwrap();

    println!("{}", arr.contiguous())

    // let arr_b = ArangeArray::arange(0..15)
    //     .to_GpuArray_with_shape(&[5, 3], &module)
    //     .unwrap();
    // println!("{}", arr_b);

    // let arr = module.add_view(&arr_a, &arr_b).unwrap();
    // println!("{}", arr);

    // println!("{:?}", module.get_heap());
}
