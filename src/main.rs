use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r};

fn main() {
    let module = ArrOgpuModule::default();

    let array_a = ArangeArray::arange(0..12).to_GpuArray(&module).unwrap();
    println!("{}", array_a);

    let array_b = ArangeArray::arange(0..12).to_GpuArray(&module).unwrap();
    println!("{}", array_b);

    let c = module
        .add_metadata(&array_a, &array_b.view().unwrap())
        .unwrap();
    println!("{}", c);
    // println!("{:?}", &module.get_heap()[38..50])
}
