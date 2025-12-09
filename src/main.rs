use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r };

fn main() {
    let module = ArrOgpuModule::default();

    let arr = ArangeArray::arange(0..3)
        .to_GpuArray_with_shape(&[1, 3], &module)
        .unwrap();
    println!("{}", arr);

    let arr = module.broadcast_view(&arr, &[2, 3, 3]).unwrap();

    println!("\n{}", arr.contiguous())
}
