use crate::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, FunctionExecuteOpt, arr_o_gpu};

#[test]
fn matmul_2d_error() {
    let module = ArrOgpuModule::default();

    let array_a = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[12], &module)
        .unwrap();
    let array_b = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[12], &module)
        .unwrap();
    let result = module.matmul(&array_a, &array_b);
    if let Ok(_) = result {
        panic!("dot product is not allowed in matmul")
    }
    drop(array_a);
    drop(array_b);

    let array_a = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[3, 4], &module)
        .unwrap();
    let array_b = ArangeArray::arange(0..12)
        .to_GpuArray_with_shape(&[2, 6], &module)
        .unwrap();
    let result = module.matmul(&array_a, &array_b);
    if let Ok(_) = result {
        panic!("the allowed shapes are [m, k] x [k, n]")
    }
    drop(array_a);
    drop(array_b);

    let array_a = ArangeArray::arange(0..24)
        .to_GpuArray_with_shape(&[24], &module)
        .unwrap();
    let array_b = ArangeArray::arange(0..240)
        .to_GpuArray_with_shape(&[24, 10], &module)
        .unwrap();
    let result = module.matmul(&array_a, &array_b);
    if let Ok(_) = result {
        panic!("shape is not the same")
    }
    drop(array_a);
    drop(array_b);
}

#[test]
fn matmul_2d_testing_a() {
    let module = ArrOgpuModule::init(crate::ArrOgpuModuleInit {
        heap_size: arr_o_gpu::HeapSize::Item(1_000_000),
        function_execute_opt: FunctionExecuteOpt {
            matmul: arr_o_gpu::MatmulOpt { workgroup_size: 16 },
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();

    // 8 x 8
    let shape = [8, 8];
    let data = (0..shape.iter().product::<u32>())
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<f32>>();
    let array_a = module.array_from_vector(&data, &shape).unwrap();
    let array_b = module.array_from_vector(&data, &shape).unwrap();
    let result = module.matmul(&array_a, &array_b).unwrap();
    let check_result = testing_matmul_2d(
        &array_a.get_heap(),
        &array_a.shape(),
        &array_b.get_heap(),
        &array_b.shape(),
    );
    assert_eq!(result.get_heap(), check_result);
    drop(array_a);
    drop(array_b);

    // 16 x 16
    let shape = [16, 16];
    let data = (0..shape.iter().product::<u32>())
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<f32>>();
    let array_a = module.array_from_vector(&data, &shape).unwrap();
    let array_b = module.array_from_vector(&data, &shape).unwrap();
    let result = module.matmul(&array_a, &array_b).unwrap();
    let check_result = testing_matmul_2d(
        &array_a.get_heap(),
        &array_a.shape(),
        &array_b.get_heap(),
        &array_b.shape(),
    );
    assert_eq!(result.get_heap(), check_result);
    drop(array_a);
    drop(array_b);

    // 32 x 32
    let shape = [32, 32];
    let data = (0..shape.iter().product::<u32>())
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<f32>>();
    let array_a = module.array_from_vector(&data, &shape).unwrap();
    let array_b = module.array_from_vector(&data, &shape).unwrap();
    let result = module.matmul(&array_a, &array_b).unwrap();
    let check_result = testing_matmul_2d(
        &array_a.get_heap(),
        &array_a.shape(),
        &array_b.get_heap(),
        &array_b.shape(),
    );
    assert_eq!(result.get_heap(), check_result);
    drop(array_a);
    drop(array_b);

    // 128 x 128
    let shape = [128, 128];
    let data = (0..shape.iter().product::<u32>())
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<f32>>();
    let array_a = module.array_from_vector(&data, &shape).unwrap();
    let array_b = module.array_from_vector(&data, &shape).unwrap();
    let result = module.matmul(&array_a, &array_b).unwrap();
    let check_result = testing_matmul_2d(
        &array_a.get_heap(),
        &array_a.shape(),
        &array_b.get_heap(),
        &array_b.shape(),
    );

    let tolerance: f32 = 1e-5;
    for (a, b) in result.get_heap().iter().zip(check_result.iter()) {
        if a != b {
            if (a - b).abs() / 1.15e9 > tolerance {
                panic!("the difference has exceeded the tolerance limit")
            }
        }
    }
    drop(array_a);
    drop(array_b);

    // 512 x 512
    let shape = [512, 512];
    let data = (0..shape.iter().product::<u32>())
        .into_iter()
        .map(|x| x as f32)
        .collect::<Vec<f32>>();
    let array_a = module.array_from_vector(&data, &shape).unwrap();
    let array_b = module.array_from_vector(&data, &shape).unwrap();
    let result = module.matmul(&array_a, &array_b).unwrap();
    let check_result = testing_matmul_2d(
        &array_a.get_heap(),
        &array_a.shape(),
        &array_b.get_heap(),
        &array_b.shape(),
    );
    let tolerance: f32 = 1e-5;
    for (a, b) in result.get_heap().iter().zip(check_result.iter()) {
        if a != b {
            if (a - b).abs() / 3.8e12 > tolerance {
                panic!("the difference has exceeded the tolerance limit")
            }
        }
    }
    drop(array_a);
    drop(array_b);
}

#[test]
fn matmul_2d_testing_b() {
    let module = ArrOgpuModule::init(crate::ArrOgpuModuleInit {
        heap_size: arr_o_gpu::HeapSize::Item(1_000_000),
        function_execute_opt: FunctionExecuteOpt {
            matmul: arr_o_gpu::MatmulOpt { workgroup_size: 16 },
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();

    // 11x23 * 23x10
    let array_a = ArangeArray::arange(0..11 * 23)
        .to_GpuArray_with_shape(&[11, 23], &module)
        .unwrap();
    let array_b = ArangeArray::arange(0..23 * 10)
        .to_GpuArray_with_shape(&[23, 10], &module)
        .unwrap();
    let result = module.matmul(&array_a, &array_b).unwrap();
    let check_result = testing_matmul_2d(
        &array_a.get_heap(),
        &array_a.shape(),
        &array_b.get_heap(),
        &array_b.shape(),
    );
    assert_eq!(result.get_heap(), check_result);
    drop(array_a);
    drop(array_b);

    // 33x47 * 47x29
    let array_a = ArangeArray::arange(0..33 * 47)
        .to_GpuArray_with_shape(&[33, 47], &module)
        .unwrap();
    let array_b = ArangeArray::arange(0..47 * 29)
        .to_GpuArray_with_shape(&[47, 29], &module)
        .unwrap();
    let result = module.matmul(&array_a, &array_b).unwrap();
    let check_result = testing_matmul_2d(
        &array_a.get_heap(),
        &array_a.shape(),
        &array_b.get_heap(),
        &array_b.shape(),
    );
    assert_eq!(result.get_heap(), check_result);
    drop(array_a);
    drop(array_b);

    // 38x41 x 41x53
    let array_a = ArangeArray::arange(0..38 * 41)
        .to_GpuArray_with_shape(&[38, 41], &module)
        .unwrap();
    let array_b = ArangeArray::arange(0..41 * 53)
        .to_GpuArray_with_shape(&[41, 53], &module)
        .unwrap();
    let result = module.matmul(&array_a, &array_b).unwrap();
    let check_result = testing_matmul_2d(
        &array_a.get_heap(),
        &array_a.shape(),
        &array_b.get_heap(),
        &array_b.shape(),
    );
    assert_eq!(result.get_heap(), check_result);
    drop(array_a);
    drop(array_b);

    // 109x111 * 111*98
    let array_a = ArangeArray::arange(0..109 * 111)
        .to_GpuArray_with_shape(&[109, 111], &module)
        .unwrap();
    let array_b = ArangeArray::arange(0..111 * 98)
        .to_GpuArray_with_shape(&[111, 98], &module)
        .unwrap();
    let result = module.matmul(&array_a, &array_b).unwrap();
    let check_result = testing_matmul_2d(
        &array_a.get_heap(),
        &array_a.shape(),
        &array_b.get_heap(),
        &array_b.shape(),
    );

    let tolerance: f32 = 1e-5;
    for (a, b) in result.get_heap().iter().zip(check_result.iter()) {
        if a != b {
            if (a - b).abs() / 1.15e9 > tolerance {
                panic!("the difference has exceeded the tolerance limit")
            }
        }
    }
    drop(array_a);
    drop(array_b);

    // 543x509 * 509x497
    let array_a = ArangeArray::arange(0..543 * 509)
        .to_GpuArray_with_shape(&[543, 509], &module)
        .unwrap();
    let array_b = ArangeArray::arange(0..509 * 497)
        .to_GpuArray_with_shape(&[509, 497], &module)
        .unwrap();
    let result = module.matmul(&array_a, &array_b).unwrap();
    let check_result = testing_matmul_2d(
        &array_a.get_heap(),
        &array_a.shape(),
        &array_b.get_heap(),
        &array_b.shape(),
    );
    let tolerance: f32 = 1e-5;
    for (a, b) in result.get_heap().iter().zip(check_result.iter()) {
        if a != b {
            if (a - b).abs() / 3.8e12 > tolerance {
                panic!("the difference has exceeded the tolerance limit")
            }
        }
    }
    drop(array_a);
    drop(array_b);
}

fn testing_matmul_2d(data_a: &[f32], shape_a: &[u32], data_b: &[f32], shape_b: &[u32]) -> Vec<f32> {
    let stride_row_a = shape_a[1];
    let stride_coll_a = 1;

    let stride_row_b = shape_b[1];
    let stride_coll_b = 1;

    let mut output = Vec::new();
    for row in 0..shape_a[0] {
        for coll in 0..shape_b[1] {
            let mut sum = 0.;
            for k in 0..shape_a[1] {
                let index_a = row * stride_row_a + k * stride_coll_a;
                let index_b = k * stride_row_b + coll * stride_coll_b;
                sum += data_a[index_a as usize] * data_b[index_b as usize];
            }
            output.push(sum);
        }
    }
    output
}
