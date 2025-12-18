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

    // let array_b = ArangeArray::arange(0..12)
    // .to_GpuArray_with_shape(&[2, 2, 3], &module)
    // .unwrap();
    // println!("{}", array_b);

    // let array_b = module.index(&array_b, &[0, 0, 2]).unwrap();

    let out = module.mul(&array_a, &5.0).unwrap();
    // println!("{}", array_b.contiguous());
    println!("{}", out);
}
