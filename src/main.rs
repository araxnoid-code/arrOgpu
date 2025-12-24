use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};

fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..20).to_GpuArray(&module).unwrap();
    let array_view = array.view();
    // let array_view = module.index(&array_view, &[1]).unwrap();

    // module.abs(&array);
    // module.abs(&array_view);

    let sum = module.sum(&array).unwrap();

    println!("{array}");
}
