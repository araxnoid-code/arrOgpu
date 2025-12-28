use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};

fn main() {
    let module = ArrOgpuModule::default();

    let array_a = ArangeArray::arange(0..6).to_GpuArray(&module).unwrap();
    let array_b = ArangeArray::arange(6..12).to_GpuArray(&module).unwrap();

    let dot = module.dot_product_optimize(&array_a, &array_b).unwrap();
    println!("{}", dot);

    let data_a = array_a.get_heap();
    let data_b = array_b.get_heap();
    let sum = data_a
        .iter()
        .zip(data_b.iter())
        .map(|(a, b)| *a * *b)
        .sum::<f32>();
    println!("{}", sum);
}
