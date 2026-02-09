use crate::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, SubOpt, r};

#[test]
fn sub_error_handler() {
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
    let result = module.sub(&array_a, &array_b).unwrap();
    drop(array_a);
    drop(array_b);
    drop(result);

    let array_a = ArangeArray::arange(0..16)
        .to_GpuArray_with_shape(&[4, 4], &module)
        .unwrap();
    let array_b = ArangeArray::arange(0..16)
        .to_GpuArray_with_shape(&[16, 1], &module)
        .unwrap();
    let result = module.sub(&array_a, &array_b);
    if let Ok(_) = result {
        panic!("this code must be error")
    }
    drop(array_a);
    drop(array_b);
    drop(result);
}

#[test]
fn sub_contiguous() {
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

    let result = module.sub(&array_a, &array_b).unwrap();

    let check = (0..256)
        .into_iter()
        .zip((256..512).into_iter())
        .map(|(a, b)| (a - b) as f32)
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

    let result = module.sub(&array_a, &array_b).unwrap();

    let check = (0..512)
        .into_iter()
        .zip((512..1024).into_iter())
        .map(|(a, b)| (a - b) as f32)
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

    let result = module.sub(&array_a, &array_b).unwrap();

    let check = (0..1024)
        .into_iter()
        .zip((1024..2048).into_iter())
        .map(|(a, b)| (a - b) as f32)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());

    drop(array_a);
    drop(array_b);
    drop(result);
}

#[test]
fn sub_view() {
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
        .sub(&array_a.view().unwrap(), &array_b.view().unwrap())
        .unwrap();

    let check = (0..256)
        .into_iter()
        .zip((256..512).into_iter())
        .map(|(a, b)| (a - b) as f32)
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
        .sub(&array_a.view().unwrap(), &array_b.view().unwrap())
        .unwrap();

    let check = (0..512)
        .into_iter()
        .zip((512..1024).into_iter())
        .map(|(a, b)| (a - b) as f32)
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

    let result = module.sub(&permute, &array_b).unwrap();

    let check = permute
        .contiguous()
        .unwrap()
        .get_heap()
        .iter()
        .zip((512..1024).into_iter())
        .map(|(a, b)| a - b as f32)
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

    let result = module.sub(&array_a, &slicing).unwrap();

    let check = array_a
        .get_heap()
        .iter()
        .zip(slicing.contiguous().unwrap().get_heap().iter())
        .map(|(a, b)| a - b)
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

    let result = module.sub(&broadcasting_a, &slicing_b).unwrap();

    let check = broadcasting_a
        .contiguous()
        .unwrap()
        .get_heap()
        .iter()
        .zip(slicing_b.contiguous().unwrap().get_heap().iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());

    drop(array_a);
    drop(array_b);
    drop(result);
}

#[test]
fn sub_scalar_contiguous() {
    let module = ArrOgpuModule::init(crate::ArrOgpuModuleInit {
        heap_size: crate::HeapSize::Item(4096),
        ..Default::default()
    })
    .unwrap();

    let array = ArangeArray::arange(0..512)
        .to_GpuArray_with_shape(&[8, 8, 8], &module)
        .unwrap();
    let scalar = 10.;

    let result = module.sub(&array, &scalar).unwrap();
    let check = (0..512)
        .into_iter()
        .map(|x| (x as f32) - scalar)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());
    drop(array);
    drop(result);

    let array = ArangeArray::arange(512..2048)
        .to_GpuArray_with_shape(&[2, 768], &module)
        .unwrap();
    let scalar = module.array_from_vector(&[20.], &[1]).unwrap();

    let result = module.sub(&array, &scalar).unwrap();
    let check = (512..2048)
        .into_iter()
        .map(|x| (x as f32) - 20.)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());
    drop(array);
    drop(result);
}

#[test]
fn sub_scalar_view() {
    let module = ArrOgpuModule::init(crate::ArrOgpuModuleInit {
        heap_size: crate::HeapSize::Item(4096),
        ..Default::default()
    })
    .unwrap();

    let array = ArangeArray::arange(0..512)
        .to_GpuArray_with_shape(&[8, 8, 8], &module)
        .unwrap();
    let scalar = 10.;

    let result = module.sub(&array.view().unwrap(), &scalar).unwrap();
    let check = (0..512)
        .into_iter()
        .map(|x| (x as f32) - scalar)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());
    drop(array);
    drop(result);

    let array = ArangeArray::arange(512..2048)
        .to_GpuArray_with_shape(&[2, 768], &module)
        .unwrap();
    let scalar = module.array_from_vector(&[20.], &[1]).unwrap();

    let result = module
        .sub(&array.view().unwrap(), &scalar.view().unwrap())
        .unwrap();
    let check = (512..2048)
        .into_iter()
        .map(|x| (x as f32) - 20.)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());
    drop(array);
    drop(result);
}

#[test]
fn sub_execute_opt_contiguous() {
    let module = ArrOgpuModule::init(crate::ArrOgpuModuleInit {
        heap_size: crate::HeapSize::Item(4096),
        limits: crate::WgpuLimits {
            max_compute_invocations_per_workgroup: 512,
            max_compute_workgroup_size_x: 512,
            ..Default::default()
        },
        function_execute_opt: crate::FunctionExecuteOpt {
            sub: SubOpt {
                compute_workgroup_size_x: 512,
            },
            ..Default::default()
        },
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

    let result = module.sub(&array_a, &array_b).unwrap();

    let check = (0..256)
        .into_iter()
        .zip((256..512).into_iter())
        .map(|(a, b)| (a - b) as f32)
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

    let sub = module.sub(&array_a, &array_b).unwrap();

    let check = (0..512)
        .into_iter()
        .zip((512..1024).into_iter())
        .map(|(a, b)| (a - b) as f32)
        .collect::<Vec<f32>>();
    assert_eq!(check, sub.get_heap());

    drop(array_a);
    drop(array_b);
    drop(sub);

    // 1024
    let array_a = ArangeArray::arange(0..1024)
        .to_GpuArray_with_shape(&[32, 32], &module)
        .unwrap();

    let array_b = ArangeArray::arange(1024..2048)
        .to_GpuArray_with_shape(&[32, 32], &module)
        .unwrap();

    let result = module.sub(&array_a, &array_b).unwrap();

    let check = (0..1024)
        .into_iter()
        .zip((1024..2048).into_iter())
        .map(|(a, b)| (a - b) as f32)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());

    drop(array_a);
    drop(array_b);
    drop(result);
}

#[test]
fn sub_execute_opt_view() {
    let module = ArrOgpuModule::init(crate::ArrOgpuModuleInit {
        heap_size: crate::HeapSize::Item(4096),
        limits: crate::WgpuLimits {
            ..Default::default()
        },
        function_execute_opt: crate::FunctionExecuteOpt {
            sub: SubOpt {
                compute_workgroup_size_x: 128,
            },
            ..Default::default()
        },
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

    let result = module.sub(&array_a, &array_b).unwrap();

    let check = (0..256)
        .into_iter()
        .zip((256..512).into_iter())
        .map(|(a, b)| (a - b) as f32)
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

    let result = module.sub(&array_a, &array_b).unwrap();

    let check = (0..512)
        .into_iter()
        .zip((512..1024).into_iter())
        .map(|(a, b)| (a - b) as f32)
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

    let result = module.sub(&array_a, &array_b).unwrap();

    let check = (0..1024)
        .into_iter()
        .zip((1024..2048).into_iter())
        .map(|(a, b)| (a - b) as f32)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());

    drop(array_a);
    drop(array_b);
    drop(result);
}

#[test]
fn sub_execute_opt_scalar_contiguous() {
    let module = ArrOgpuModule::init(crate::ArrOgpuModuleInit {
        heap_size: crate::HeapSize::Item(4096),
        limits: crate::WgpuLimits {
            max_compute_invocations_per_workgroup: 512,
            max_compute_workgroup_size_x: 512,
            ..Default::default()
        },
        function_execute_opt: crate::FunctionExecuteOpt {
            sub: SubOpt {
                compute_workgroup_size_x: 512,
            },
            ..Default::default()
        },
        ..Default::default()
    })
    .expect("module init in sub error");

    let array = ArangeArray::arange(0..512)
        .to_GpuArray_with_shape(&[8, 8, 8], &module)
        .unwrap();
    let scalar = 10.;

    let result = module.sub(&array, &scalar).unwrap();
    let check = (0..512)
        .into_iter()
        .map(|x| (x as f32) - scalar)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());
    drop(array);
    drop(result);

    let array = ArangeArray::arange(512..2048)
        .to_GpuArray_with_shape(&[2, 768], &module)
        .unwrap();
    let scalar = module.array_from_vector(&[20.], &[1]).unwrap();

    let result = module.sub(&array, &scalar).unwrap();
    let check = (512..2048)
        .into_iter()
        .map(|x| (x as f32) - 20.)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());
    drop(array);
    drop(result);
}

#[test]
fn sub_execute_opt_scalar_view() {
    let module = ArrOgpuModule::init(crate::ArrOgpuModuleInit {
        heap_size: crate::HeapSize::Item(4096),
        limits: crate::WgpuLimits {
            ..Default::default()
        },
        function_execute_opt: crate::FunctionExecuteOpt {
            sub: SubOpt {
                compute_workgroup_size_x: 64,
            },
            ..Default::default()
        },
        ..Default::default()
    })
    .expect("module init in sub error");

    let array = ArangeArray::arange(0..512)
        .to_GpuArray_with_shape(&[8, 8, 8], &module)
        .unwrap();
    let scalar = 10.;

    let result = module.sub(&array.view().unwrap(), &scalar).unwrap();
    let check = (0..512)
        .into_iter()
        .map(|x| (x as f32) - scalar)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());
    drop(array);
    drop(result);

    let array = ArangeArray::arange(512..2048)
        .to_GpuArray_with_shape(&[2, 768], &module)
        .unwrap();
    let scalar = module.array_from_vector(&[20.], &[1]).unwrap();

    let result = module
        .sub(&array.view().unwrap(), &scalar.view().unwrap())
        .unwrap();
    let check = (512..2048)
        .into_iter()
        .map(|x| (x as f32) - 20.)
        .collect::<Vec<f32>>();
    assert_eq!(check, result.get_heap());
    drop(array);
    drop(result);
}
