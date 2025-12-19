use arr_o_gpu::{
    ArangeArray,
    ArangeIteratorTrait,
    ArrOgpuErr,
    ArrOgpuModule,
    ArrayView,
    GpuArray,
    r,
};

fn main() {
    let module = ArrOgpuModule::default();

    let array_a = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[2, 2, 3], &module)
        .unwrap();
    println!("{}", array_a);

    let out = module.tan(&array_a).unwrap();

    println!("{}", out);
}
