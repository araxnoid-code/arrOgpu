use crate::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, PowTrait, r};

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

#[test]
fn add_basic_view() {
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

    let sum = module
        .add(&array_a.view().unwrap(), &array_b.view().unwrap())
        .unwrap();

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

    let sum = module
        .add(&array_a.view().unwrap(), &array_b.view().unwrap())
        .unwrap();

    let sum_manual = (0..512)
        .into_iter()
        .zip((512..1024).into_iter())
        .map(|(a, b)| (a + b) as f32)
        .collect::<Vec<f32>>();
    assert_eq!(sum_manual, sum.get_heap());

    drop(array_a);
    drop(array_b);
    drop(sum);

    let array_a = ArangeArray::arange(0..512)
        .to_GpuArray_with_shape(&[8, 8, 8], &module)
        .unwrap();
    let permute = module.permute(&array_a, &[2, 1, 0]).unwrap();

    let array_b = ArangeArray::arange(512..1024)
        .to_GpuArray_with_shape(&[8, 8, 8], &module)
        .unwrap();

    let sum = module.add(&permute, &array_b).unwrap();

    let sum_manual = permute
        .contiguous()
        .unwrap()
        .get_heap()
        .iter()
        .zip((512..1024).into_iter())
        .map(|(a, b)| a + b as f32)
        .collect::<Vec<f32>>();
    assert_eq!(sum_manual, sum.get_heap());

    drop(array_a);
    drop(array_b);
    drop(sum);

    let array_a = ArangeArray::arange(0..256)
        .to_GpuArray_with_shape(&[16, 16], &module)
        .unwrap();

    let array_b = ArangeArray::arange(0..1024)
        .to_GpuArray_with_shape(&[32, 32], &module)
        .unwrap();
    let slicing = module.slicing(&array_b, &[r(16..), r(..16)]).unwrap();

    let sum = module.add(&array_a, &slicing).unwrap();

    let sum_manual = array_a
        .get_heap()
        .iter()
        .zip(slicing.contiguous().unwrap().get_heap().iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<f32>>();
    assert_eq!(sum_manual, sum.get_heap());

    drop(array_a);
    drop(array_b);
    drop(sum);

    let array_a = ArangeArray::arange(0..256)
        .to_GpuArray_with_shape(&[16, 16], &module)
        .unwrap();
    let indexing_a = array_a.index(&[8]).unwrap();
    let broadcasting_a = module.broadcast(&indexing_a, &[16, 16]).unwrap();

    let array_b = ArangeArray::arange(0..1024)
        .to_GpuArray_with_shape(&[32, 32], &module)
        .unwrap();
    let permute_b = module.permute(&array_b, &[1, 0]).unwrap();
    let slicing_b = module.slicing(&permute_b, &[r(..16), r(16..)]).unwrap();

    let sum = module.add(&broadcasting_a, &slicing_b).unwrap();

    let manual_sum = broadcasting_a
        .contiguous()
        .unwrap()
        .get_heap()
        .iter()
        .zip(slicing_b.contiguous().unwrap().get_heap().iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<f32>>();
    assert_eq!(manual_sum, sum.get_heap());

    drop(array_a);
    drop(array_b);
    drop(sum);
}
