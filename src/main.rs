use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r};

fn main() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        heap_size: arr_o_gpu::HeapSize::Item(2000000),
        ..Default::default()
    })
    .unwrap();

    let data = [0.3; 409600];
    let array_a = module.array_from_vector(&data, &[409600]).unwrap();
    let array_b = module.array_from_vector(&data, &[409600]).unwrap();

    let dot = array_a
        .get_heap()
        .iter()
        .zip(array_b.get_heap().iter())
        .map(|(a, b)| a * b)
        .sum::<f32>();
    println!("{}", dot);
    // let dot = module.dot_product(&array_a, &array_b).unwrap();
    // println!("{}", dot);
    // let dot_heap = dot.get_heap();

    let dot_unsave = module.dot_product_unsave(&array_a, &array_b).unwrap();
    println!("{}", dot_unsave);
    // let dot_unsave_heap = dot_unsave.get_heap();
    // println!("{:?}", &module.get_heap()[1601..2000]);

    // println!("{}", dot_heap == dot_unsave_heap);

    // let array_b = ArangeArray::arange(20..45)
    //     .to_GpuArray_with_shape(&[5, 5], &module)
    //     .unwrap();

    // let result_a = module.matmul(&view, &array_b).unwrap();
    // println!("{}", result_a);

    // let contiguous = view.contiguous_metadata().unwrap();
    // let result_b = module.matmul(&contiguous, &array_b).unwrap();
    // println!("{}", result_b);

    // println!("{}", result_a.get_heap() == result_b.get_heap());
}
