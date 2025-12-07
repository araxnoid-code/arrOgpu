use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule };

fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..24)
        .to_GpuArray_with_shape(&[3, 2, 4], &module)
        .unwrap();

    let array_view = module.index_view(&array, &[0]).unwrap().collect();

    // println!("{:?}", module.get_heap());
}
