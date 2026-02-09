use crate::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, r};

#[test]
fn div_error_handler() {
    let module = ArrOgpuModule::init(crate::ArrOgpuModuleInit {
        ..Default::default()
    })
    .unwrap();

    let array_a = ArangeArray::arange(0..16)
        .to_GpuArray_with_shape(&[4, 4], &module)
        .unwrap();
    let array_b = ArangeArray::arange(0..16)
        .to_GpuArray_with_shape(&[4, 4], &module)
        .unwrap();
    let result = module.div(&array_a, &array_b).unwrap();
    drop(array_a);
    drop(array_b);
    drop(result);

    let array_a = ArangeArray::arange(0..16)
        .to_GpuArray_with_shape(&[4, 4], &module)
        .unwrap();
    let array_b = ArangeArray::arange(0..16)
        .to_GpuArray_with_shape(&[16, 1], &module)
        .unwrap();
    let result = module.div(&array_a, &array_b);
    if let Ok(_) = result {
        panic!("this code must be error")
    }
    drop(array_a);
    drop(array_b);
    drop(result);
}

#[test]
fn div_contiguous() {
    let module = ArrOgpuModule::init(crate::ArrOgpuModuleInit {
        heap_size: crate::HeapSize::Item(4096),
        ..Default::default()
    })
    .expect("module init in sub error");

    // 256
    let array_a = ArangeArray::arange(0..256)
        .to_GpuArray_with_shape(&[16, 16], &module)
        .unwrap();

    let array_b = ArangeArray::arange(256..512)
        .to_GpuArray_with_shape(&[16, 16], &module)
        .unwrap();

    let result = module.div(&array_a, &array_b).unwrap();

    let check = (0..256)
        .into_iter()
        .zip((256..512).into_iter())
        .map(|(a, b)| a as f32 / b as f32)
        .collect::<Vec<f32>>();

    assert_eq!(check, result.get_heap());

    drop(array_a);
    drop(array_b);
    drop(result);

    // 512
    let array_a = ArangeArray::arange(0..512)
        .to_GpuArray_with_shape(&[8, 8, 8], &module)
        .unwrap();

    let array_b = ArangeArray::arange(512..1024)
        .to_GpuArray_with_shape(&[8, 8, 8], &module)
        .unwrap();

    let result = module.div(&array_a, &array_b).unwrap();

    let check = (0..512)
        .into_iter()
        .zip((512..1024).into_iter())
        .map(|(a, b)| a as f32 / b as f32)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());

    drop(array_a);
    drop(array_b);
    drop(result);

    // 1024
    let array_a = ArangeArray::arange(0..1024)
        .to_GpuArray_with_shape(&[32, 32], &module)
        .unwrap();

    let array_b = ArangeArray::arange(1024..2048)
        .to_GpuArray_with_shape(&[32, 32], &module)
        .unwrap();

    let result = module.div(&array_a, &array_b).unwrap();

    let check = (0..1024)
        .into_iter()
        .zip((1024..2048).into_iter())
        .map(|(a, b)| (a / b) as f32)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());

    drop(array_a);
    drop(array_b);
    drop(result);
}

#[test]
fn div_view() {
    let module = ArrOgpuModule::init(crate::ArrOgpuModuleInit {
        heap_size: crate::HeapSize::Item(4096),
        ..Default::default()
    })
    .expect("module init in sub error");

    // 256
    let array_a = ArangeArray::arange(0..256)
        .to_GpuArray_with_shape(&[16, 16], &module)
        .unwrap();

    let array_b = ArangeArray::arange(256..512)
        .to_GpuArray_with_shape(&[16, 16], &module)
        .unwrap();

    let result = module
        .div(&array_a.view().unwrap(), &array_b.view().unwrap())
        .unwrap();

    let check = (0..256)
        .into_iter()
        .zip((256..512).into_iter())
        .map(|(a, b)| a as f32 / b as f32)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());

    drop(array_a);
    drop(array_b);
    drop(result);

    // 512
    let array_a = ArangeArray::arange(0..512)
        .to_GpuArray_with_shape(&[8, 8, 8], &module)
        .unwrap();

    let array_b = ArangeArray::arange(512..1024)
        .to_GpuArray_with_shape(&[8, 8, 8], &module)
        .unwrap();

    let result = module
        .div(&array_a.view().unwrap(), &array_b.view().unwrap())
        .unwrap();

    let check = (0..512)
        .into_iter()
        .zip((512..1024).into_iter())
        .map(|(a, b)| (a / b) as f32)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());

    drop(array_a);
    drop(array_b);
    drop(result);

    let array_a = ArangeArray::arange(0..512)
        .to_GpuArray_with_shape(&[8, 8, 8], &module)
        .unwrap();
    let permute = module.permute(&array_a, &[2, 1, 0]).unwrap();

    let array_b = ArangeArray::arange(512..1024)
        .to_GpuArray_with_shape(&[8, 8, 8], &module)
        .unwrap();

    let result = module.div(&permute, &array_b).unwrap();

    let check = permute
        .contiguous()
        .unwrap()
        .get_heap()
        .iter()
        .zip((512..1024).into_iter())
        .map(|(a, b)| a / b as f32)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());

    drop(array_a);
    drop(array_b);
    drop(result);

    let array_a = ArangeArray::arange(0..256)
        .to_GpuArray_with_shape(&[16, 16], &module)
        .unwrap();

    let array_b = ArangeArray::arange(0..1024)
        .to_GpuArray_with_shape(&[32, 32], &module)
        .unwrap();
    let slicing = module.slicing(&array_b, &[r(16..), r(..16)]).unwrap();

    let result = module.div(&array_a, &slicing).unwrap();

    let check = array_a
        .get_heap()
        .iter()
        .zip(slicing.contiguous().unwrap().get_heap().iter())
        .map(|(a, b)| a / b)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());

    drop(array_a);
    drop(array_b);
    drop(result);

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

    let result = module.div(&broadcasting_a, &slicing_b).unwrap();

    let check = broadcasting_a
        .contiguous()
        .unwrap()
        .get_heap()
        .iter()
        .zip(slicing_b.contiguous().unwrap().get_heap().iter())
        .map(|(a, b)| a / b)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());

    drop(array_a);
    drop(array_b);
    drop(result);
}
