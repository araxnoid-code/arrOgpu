use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r };

fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..24)
        .to_GpuArray_with_shape(&[3, 2, 4], &module)
        .unwrap();
    print!("{}", array);

    let arr = module
        .slicing_view(&array, &[r(0..2), r(0..1)])
        .unwrap()
        .collect();

    println!("\n{}", arr);

    println!("\n{:?}", module.get_heap())
}
