use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r };

fn main() {
    let module = ArrOgpuModule::default();

    let arr = ArangeArray::arange(0..16)
        .to_GpuArray_with_shape(&[2, 2, 4], &module)
        .unwrap();
    println!("{}", arr);

    let arr = module.permute(&arr, &[2, 1, 1]).unwrap();
    // let arr = module.slicing_view(&arr, &[r(1..3)]).unwrap();
    // let arr = module.index_view(&arr, &[1, 1]).unwrap();
    // let arr = module.index_view(&arr, &[0]).unwrap();

    // let arr = module.permute(&arr, &[1, 0]).unwrap();

    // let binding = [r(0..2), r(0..1)];
    // let arr = module.reshape(&array, &[2, 3, 4]).unwrap();
    // let arr = module.slicing_view(&arr, &[r(..), r(1..), r(1..3)]).unwrap();

    // let arr = module.reshape(&arr, &[3, 2]).unwrap();
    // let arr = module.slicing_view(&arr, &[r(1..)]).unwrap();
    // let arr = module.index_view(&arr, &[1, 0]).unwrap();

    // let arr = module.index_view(&arr, &[]);

    // println!("\n{:?}", arr.get_heap())
}
