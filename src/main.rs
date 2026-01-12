use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r};

fn main() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        heap_size: arr_o_gpu::HeapSize::Item(2000000),
        ..Default::default()
    })
    .unwrap();

    let array = ArangeArray::arange(0..1200)
        .to_GpuArray_with_shape(&[3, 20, 20], &module)
        .unwrap();

    let view = array.index(&[1]).unwrap();
    let view = view.slicing(&[r(5..10), r(5..10)]).unwrap();
    let view = view.permute(&[1, 0]).unwrap();

    let array_b = ArangeArray::arange(20..45)
        .to_GpuArray_with_shape(&[5, 5], &module)
        .unwrap();

    let result_a = module.matmul(&view, &array_b).unwrap();
    println!("{}", result_a);

    let contiguous = view.contiguous_metadata().unwrap();
    let result_b = module.matmul(&contiguous, &array_b).unwrap();
    println!("{}", result_b);

    println!("{}", result_a.get_heap() == result_b.get_heap());
}
