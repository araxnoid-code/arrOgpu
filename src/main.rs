use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r};

fn main() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        heap_size: arr_o_gpu::HeapSize::Item(2_000_000),
        ..Default::default()
    })
    .unwrap();

    let array_a = ArangeArray::arange(0..65536)
        .to_GpuArray_with_shape(&[256, 256], &module)
        .unwrap();
    let array_b = ArangeArray::arange(65536..65536 * 2)
        .to_GpuArray_with_shape(&[256, 128, 2], &module)
        .unwrap();

    let view_a = module.permute(&array_a, &[1, 0]).unwrap();
    let view_a = module.slicing(&view_a, &[r(..64), r(32..64)]).unwrap();
    let view_a = module.index(&view_a, &[32]).unwrap();
    let view_a = module.slicing(&view_a, &[r(16..)]).unwrap();
    // println!("{}", view_a.contiguous().unwrap());

    let view_b = module.to_shape(&array_b, &[128, 2, 256]).unwrap();
    let view_b = module
        .slicing(&view_b, &[r(..), r(1..), r(32..64)])
        .unwrap();
    let view_b = module.broadcast(&view_b, &[128, 4, 32]).unwrap();
    let view_b = module.permute(&view_b, &[2, 0, 1]).unwrap();
    let view_b = module.permute(&view_b, &[1, 0, 2]).unwrap();
    let view_b = module.permute(&view_b, &[2, 0, 1]).unwrap();
    let view_b = module.index(&view_b, &[2, 16]).unwrap();
    let view_b = module.slicing(&view_b, &[r(..16)]).unwrap();
    // println!("{}", view_b.contiguous().unwrap());

    let result = module.dot_product(&view_a, &view_b).unwrap();
    println!("{}", result);

    let dot = view_a
        .contiguous()
        .unwrap()
        .get_heap()
        .iter()
        .zip(view_b.contiguous().unwrap().get_heap().iter())
        .map(|(a, b)| a * b)
        .sum::<f32>();
    println!("{}", dot);
}
