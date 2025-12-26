use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, GpuArray};

fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..12).to_GpuArray(&module).unwrap();
    let view = array.view();

    let arr = array.log2();
}
