use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r };

fn main() {
    let module = ArrOgpuModule::default();

    let arr_a = ArangeArray::arange(0..15)
        .to_GpuArray_with_shape(&[5, 3], &module)
        .unwrap();
    println!("{}", arr_a);

    let arr_b = ArangeArray::arange(0..15)
        .to_GpuArray_with_shape(&[5, 3], &module)
        .unwrap();
    println!("{}", arr_b);

    let arr = module.add_view(&arr_a, &arr_b).unwrap();
    println!("{}", arr);

    // println!("{:?}", module.get_heap());
}
