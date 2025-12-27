use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, GpuArray, r};

fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[4, 3], &module)
        .unwrap();

    println!("{}", array);

    let index = array.slicing(&[r(0..-2), r(-2..)]).unwrap();

    println!("{}", index.contiguous());
}
