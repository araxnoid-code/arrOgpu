use crate::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};

#[test]
fn add_basic_contiguous() {
    let module = ArrOgpuModule::init(crate::ArrOgpuModuleInit {
        heap_size: crate::HeapSize::Item(4096),
        ..Default::default()
    })
    .expect("module init in add_basic error");

    // 256
    let array_a = ArangeArray::arange(0..256)
        .to_GpuArray_with_shape(&[16, 16], &module)
        .unwrap();

    let array_b = ArangeArray::arange(256..512)
        .to_GpuArray_with_shape(&[16, 16], &module)
        .unwrap();

    let sum = module.add(&array_a, &array_b).unwrap();

    let sum_manual = (0..256)
        .into_iter()
        .zip((256..512).into_iter())
        .map(|(a, b)| (a + b) as f32)
        .collect::<Vec<f32>>();
    assert_eq!(sum_manual, sum.get_heap());

    drop(array_a);
    drop(array_b);
    drop(sum);

    // 512
    let array_a = ArangeArray::arange(0..512)
        .to_GpuArray_with_shape(&[8, 8, 8], &module)
        .unwrap();

    let array_b = ArangeArray::arange(512..1024)
        .to_GpuArray_with_shape(&[8, 8, 8], &module)
        .unwrap();

    let sum = module.add(&array_a, &array_b).unwrap();

    let sum_manual = (0..512)
        .into_iter()
        .zip((512..1024).into_iter())
        .map(|(a, b)| (a + b) as f32)
        .collect::<Vec<f32>>();
    assert_eq!(sum_manual, sum.get_heap());

    drop(array_a);
    drop(array_b);
    drop(sum);

    // 1024
    let array_a = ArangeArray::arange(0..1024)
        .to_GpuArray_with_shape(&[32, 32], &module)
        .unwrap();

    let array_b = ArangeArray::arange(1024..2048)
        .to_GpuArray_with_shape(&[32, 32], &module)
        .unwrap();

    let sum = module.add(&array_a, &array_b).unwrap();

    let sum_manual = (0..1024)
        .into_iter()
        .zip((1024..2048).into_iter())
        .map(|(a, b)| (a + b) as f32)
        .collect::<Vec<f32>>();
    assert_eq!(sum_manual, sum.get_heap());

    drop(array_a);
    drop(array_b);
    drop(sum);
}
