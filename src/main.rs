use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};

fn main() {
    let module = ArrOgpuModule::default();

    let array = ArangeArray::arange(0..24)
        .to_GpuArray_with_shape(&[2, 3, 4], &module)
        .unwrap();
    println!("{}", array);

    let sum_axis = module.sum_axis_unsafe(&array, &[0, 1]).unwrap();
    println!("{}", sum_axis);

    // println!("{:?}", &module.get_heap()[24..100]);
}
