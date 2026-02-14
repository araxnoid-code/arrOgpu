use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r};

fn main() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        heap_size: arr_o_gpu::HeapSize::Item(1000000),
        ..Default::default()
    })
    .unwrap();

    // testing 0
    let array = ArangeArray::arange(0..8192)
        .to_GpuArray_with_shape(&[8, 32, 32], &module)
        .unwrap();

    let slicing = module.slicing(&array, &[r(..), r(16..), r(..16)]).unwrap();
    let view_result = module.matmul(&slicing, &slicing).unwrap();
    let slicing_heap = view_result.get_heap();
    let contiguous = slicing.contiguous().unwrap();
    let contiguous_result = module.matmul(&contiguous, &contiguous).unwrap();
    assert_eq!(slicing_heap, contiguous_result.get_heap());
    drop(array);
    drop(contiguous);
    drop(contiguous_result);

    // testing 1
    let array = ArangeArray::arange(0..3072)
        .to_GpuArray_with_shape(&[3, 16, 8, 8], &module)
        .unwrap();

    let slicing = module.slicing(&array, &[r(..), r(4..12)]).unwrap();
    let view_result = module.matmul(&slicing, &slicing).unwrap();
    let slicing_heap = view_result.get_heap();
    let contiguous = slicing.contiguous().unwrap();
    let contiguous_result = module.matmul(&contiguous, &contiguous).unwrap();
    assert_eq!(slicing_heap, contiguous_result.get_heap());
    drop(array);
    drop(contiguous);
    drop(contiguous_result);

    // testing 2
    let array = ArangeArray::arange(0..30720)
        .to_GpuArray_with_shape(&[3, 16, 10, 8, 8], &module)
        .unwrap();

    let slicing = module.slicing(&array, &[r(..), r(4..12)]).unwrap();
    let view_result = module.matmul(&slicing, &slicing).unwrap();
    let slicing_heap = view_result.get_heap();
    let contiguous = slicing.contiguous().unwrap();
    let contiguous_result = module.matmul(&contiguous, &contiguous).unwrap();
    assert_eq!(slicing_heap, contiguous_result.get_heap());
    drop(array);
    drop(contiguous);
    drop(contiguous_result);

    // testing 3
    let array = ArangeArray::arange(0..36864)
        .to_GpuArray_with_shape(&[3, 16, 12, 8, 8], &module)
        .unwrap();

    let slicing = module.slicing(&array, &[r(..), r(4..12), r(3..9)]).unwrap();
    let view_result = module.matmul(&slicing, &slicing).unwrap();
    let slicing_heap = view_result.get_heap();
    let contiguous = slicing.contiguous().unwrap();
    let contiguous_result = module.matmul(&contiguous, &contiguous).unwrap();
    assert_eq!(slicing_heap, contiguous_result.get_heap());
    drop(array);
    drop(contiguous);
    drop(contiguous_result);
}
