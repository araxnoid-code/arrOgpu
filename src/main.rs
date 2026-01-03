use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, ArrayCompute, r};

fn main() {
    let module = ArrOgpuModule::default();

    let array_a = ArangeArray::arange(0..12).to_GpuArray(&module).unwrap();
    println!("{}", array_a);

    let array_b = ArangeArray::arange(0..12).to_GpuArray(&module).unwrap();
    println!("{}", array_b);

    let view = array_a.view().unwrap();

    let c = module.add_metadata(&view, &array_b).unwrap();
    println!("{}", c);
    println!("{:?}", &module.get_heap()[37..75])
}
