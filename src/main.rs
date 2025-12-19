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

    let array_a = ArangeArray::arange(0..120)
        .to_GpuArray_with_shape(&[20, 2, 3], &module)
        .unwrap();
    println!("{}", array_a);

    let dim = vec![1];

    let out_keep_dim = module.sum(&array_a).unwrap();
    println!("{}", out_keep_dim);
}
