use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, ArrayCompute, r};

fn main() {
    let module = ArrOgpuModule::default();

    let array_a = ArangeArray::arange(0..12).to_GpuArray(&module).unwrap();
    println!("{}", array_a);
    // let view = array_a.view().unwrap();

    let array_b = ArangeArray::arange(12..24).to_GpuArray(&module).unwrap();
    println!("{}", array_b);

    let c = module.add(&array_b, &array_a).unwrap();
    // println!("{}", view.contiguous());
    println!("{}", c);
}
