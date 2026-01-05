use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};

fn main() {
    let module = ArrOgpuModule::default();

    let array_a = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[3, 4], &module)
        .unwrap();
    println!("array a");
    println!("{}", array_a);

    let array_b = ArangeArray::arange(12..24)
        .to_GpuArray_with_shape(&[3, 4], &module)
        .unwrap();
    println!("array b");
    println!("{}", array_b);

    let result = array_a.sub(&array_b).unwrap();
    println!("result");
    println!("{}", result);

    let array_c = ArangeArray::arange(20..44)
        .to_GpuArray_with_shape(&[2, 3, 4], &module)
        .unwrap();
    println!("array c");
    println!("{}", array_c);

    let array_d = ArangeArray::arange(100..124)
        .to_GpuArray_with_shape(&[2, 3, 4], &module)
        .unwrap();
    println!("array d");
    println!("{}", array_d);

    let result = array_c.sub(&array_d).unwrap();
    println!("result");
    println!("{}", result);
}
