use crate::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, arr_o_gpu};

#[test]
fn matmul_nd_error() {
    let module = ArrOgpuModule::default();

    let array_a = ArangeArray::arange(0..24)
        .to_GpuArray_with_shape(&[2, 3, 4], &module)
        .unwrap();

    let array_b = ArangeArray::arange(0..40)
        .to_GpuArray_with_shape(&[2, 5, 4], &module)
        .unwrap();

    let result = module.matmul(&array_a, &array_b);
    if let Ok(_) = result {
        panic!("error, valid input for matmul_nd should be [.., m, k] * [.., k, n]")
    }
    drop(array_a);
    drop(array_b);

    let array_a = ArangeArray::arange(0..32)
        .to_GpuArray_with_shape(&[2, 2, 2, 4], &module)
        .unwrap();

    let array_b = ArangeArray::arange(0..120)
        .to_GpuArray_with_shape(&[3, 2, 4, 5], &module)
        .unwrap();

    let result = module.matmul(&array_a, &array_b);
    if let Ok(_) = result {
        panic!(
            "error, valid input for matmul_nd should be [.., m, k] * [.., k, n] where the dimension before m and the dimension before k must be the same"
        )
    }
}

#[test]
fn matmul_nd_testing_contiguous() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        heap_size: arr_o_gpu::HeapSize::Item(1_000_000),

        ..Default::default()
    })
    .unwrap();

    // testing 1
    let array_a = ArangeArray::arange(0..192)
        .to_GpuArray_with_shape(&[3, 8, 8], &module)
        .unwrap();

    let array_b = ArangeArray::arange(192..192 * 2)
        .to_GpuArray_with_shape(&[3, 8, 8], &module)
        .unwrap();

    let result = module.matmul(&array_a, &array_b).unwrap();

    let check = matmul_nd_checker(
        &array_a.get_heap(),
        &array_b.get_heap(),
        &array_a.shape(),
        &array_b.shape(),
    );
    assert_eq!(check, result.get_heap());
    drop(array_a);
    drop(array_b);

    // testing 2
    let array_a = ArangeArray::arange(0..1024)
        .to_GpuArray_with_shape(&[2, 32, 16], &module)
        .unwrap();

    let array_b = ArangeArray::arange(1024..1024 * 2)
        .to_GpuArray_with_shape(&[2, 16, 32], &module)
        .unwrap();

    let result = module.matmul(&array_a, &array_b).unwrap();

    let check = matmul_nd_checker(
        &array_a.get_heap(),
        &array_b.get_heap(),
        &array_a.shape(),
        &array_b.shape(),
    );
    assert_eq!(check, result.get_heap());
    drop(array_a);
    drop(array_b);

    // testing 3
    let array_a = ArangeArray::arange(0..37422)
        .to_GpuArray_with_shape(&[2, 189, 99], &module)
        .unwrap();

    let array_b = ArangeArray::arange(7920..7920 * 2)
        .to_GpuArray_with_shape(&[2, 99, 40], &module)
        .unwrap();

    let result = module.matmul(&array_a, &array_b).unwrap();

    let check = matmul_nd_checker(
        &array_a.get_heap(),
        &array_b.get_heap(),
        &array_a.shape(),
        &array_b.shape(),
    );

    result
        .get_heap()
        .iter()
        .zip(check.iter())
        .for_each(|(a, b)| {
            let error = (a - b).abs() / a.max(*b);
            if error > 0.00001 {
                panic!("error exceeding tolerance")
            }
        });
    drop(array_a);
    drop(array_b);

    // testing 4
    let array_a = ArangeArray::arange(0..1536)
        .to_GpuArray_with_shape(&[2, 3, 16, 16], &module)
        .unwrap();

    let array_b = ArangeArray::arange(1536..1536 * 2)
        .to_GpuArray_with_shape(&[2, 3, 16, 16], &module)
        .unwrap();

    let result = module.matmul(&array_a, &array_b).unwrap();

    let check = matmul_nd_checker(
        &array_a.get_heap(),
        &array_b.get_heap(),
        &array_a.shape(),
        &array_b.shape(),
    );
    assert_eq!(check, result.get_heap());
    drop(array_a);
    drop(array_b);

    // testing 4
    let array_a = ArangeArray::arange(0..3584)
        .to_GpuArray_with_shape(&[4, 2, 14, 32], &module)
        .unwrap();

    let array_b = ArangeArray::arange(4608..4608 * 2)
        .to_GpuArray_with_shape(&[4, 2, 32, 18], &module)
        .unwrap();

    let result = module.matmul(&array_a, &array_b).unwrap();

    let check = matmul_nd_checker(
        &array_a.get_heap(),
        &array_b.get_heap(),
        &array_a.shape(),
        &array_b.shape(),
    );
    result
        .get_heap()
        .iter()
        .zip(check.iter())
        .for_each(|(a, b)| {
            let error = (a - b).abs() / a.max(*b);
            if error > 0.00001 {
                panic!("error exceeding tolerance")
            }
        });
    drop(array_a);
    drop(array_b);
}

#[test]
fn matmul_nd_testing_view() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        heap_size: arr_o_gpu::HeapSize::Item(1000000),

        ..Default::default()
    })
    .unwrap();

    // testing 1
    let array_a = ArangeArray::arange(0..8)
        .to_GpuArray_with_shape(&[2, 2, 2], &module)
        .unwrap();

    let array_b = ArangeArray::arange(8..8 * 2)
        .to_GpuArray_with_shape(&[2, 2, 2], &module)
        .unwrap();

    let result = module
        .matmul(&array_a.view().unwrap(), &array_b.view().unwrap())
        .unwrap();

    let check = matmul_nd_checker(
        &array_a.get_heap(),
        &array_b.get_heap(),
        &array_a.shape(),
        &array_b.shape(),
    );

    assert_eq!(check, result.get_heap());
    drop(array_a);
    drop(array_b);

    // testing 2
    let array_a = ArangeArray::arange(0..1024)
        .to_GpuArray_with_shape(&[2, 32, 16], &module)
        .unwrap();

    let array_b = ArangeArray::arange(1024..1024 * 2)
        .to_GpuArray_with_shape(&[2, 16, 32], &module)
        .unwrap();

    let result = module
        .matmul(&array_a.view().unwrap(), &array_b.view().unwrap())
        .unwrap();

    let check = matmul_nd_checker(
        &array_a.get_heap(),
        &array_b.get_heap(),
        &array_a.shape(),
        &array_b.shape(),
    );
    assert_eq!(check, result.get_heap());
    drop(array_a);
    drop(array_b);

    // testing 3
    let array_a = ArangeArray::arange(0..37422)
        .to_GpuArray_with_shape(&[2, 189, 99], &module)
        .unwrap();

    let array_b = ArangeArray::arange(7920..7920 * 2)
        .to_GpuArray_with_shape(&[2, 99, 40], &module)
        .unwrap();

    let result = module
        .matmul(&array_a.view().unwrap(), &array_b.view().unwrap())
        .unwrap();

    let check = matmul_nd_checker(
        &array_a.get_heap(),
        &array_b.get_heap(),
        &array_a.shape(),
        &array_b.shape(),
    );

    result
        .get_heap()
        .iter()
        .zip(check.iter())
        .for_each(|(a, b)| {
            let error = (a - b).abs() / a.max(*b);
            if error > 0.00001 {
                panic!("error exceeding tolerance")
            }
        });
    drop(array_a);
    drop(array_b);

    // testing 4
    let array_a = ArangeArray::arange(0..1536)
        .to_GpuArray_with_shape(&[2, 3, 16, 16], &module)
        .unwrap();

    let array_b = ArangeArray::arange(1536..1536 * 2)
        .to_GpuArray_with_shape(&[2, 3, 16, 16], &module)
        .unwrap();

    let result = module
        .matmul(&array_a.view().unwrap(), &array_b.view().unwrap())
        .unwrap();

    let check = matmul_nd_checker(
        &array_a.get_heap(),
        &array_b.get_heap(),
        &array_a.shape(),
        &array_b.shape(),
    );
    assert_eq!(check, result.get_heap());
    drop(array_a);
    drop(array_b);

    // testing 4
    let array_a = ArangeArray::arange(0..3584)
        .to_GpuArray_with_shape(&[4, 2, 14, 32], &module)
        .unwrap();

    let array_b = ArangeArray::arange(4608..4608 * 2)
        .to_GpuArray_with_shape(&[4, 2, 32, 18], &module)
        .unwrap();

    let result = module
        .matmul(&array_a.view().unwrap(), &array_b.view().unwrap())
        .unwrap();

    let check = matmul_nd_checker(
        &array_a.get_heap(),
        &array_b.get_heap(),
        &array_a.shape(),
        &array_b.shape(),
    );
    result
        .get_heap()
        .iter()
        .zip(check.iter())
        .for_each(|(a, b)| {
            let error = (a - b).abs() / a.max(*b);
            if error > 0.00001 {
                panic!("error exceeding tolerance")
            }
        });
    drop(array_a);
    drop(array_b);
}

fn matmul_nd_checker(data_a: &[f32], data_b: &[f32], shape_a: &[u32], shape_b: &[u32]) -> Vec<f32> {
    let stride_a = shape_a
        .iter()
        .enumerate()
        .map(|(i, _)| shape_a[i + 1..].iter().product::<u32>())
        .collect::<Vec<u32>>();

    let stride_b = shape_b
        .iter()
        .enumerate()
        .map(|(i, _)| shape_b[i + 1..].iter().product::<u32>())
        .collect::<Vec<u32>>();

    let jump_a = stride_a[stride_a.len() - 3];
    let jump_b = stride_b[stride_b.len() - 3];
    let j = shape_a[..shape_a.len() - 2].iter().product::<u32>();
    let m = shape_a[shape_a.len() - 2];
    let k = shape_a[shape_a.len() - 1];
    let n = shape_b[shape_b.len() - 1];

    let mut output = Vec::new();
    for j in 0..j {
        for row in 0..m {
            for coll in 0..n {
                let mut sum = 0.;
                for i in 0..k {
                    let index_a = stride_a[stride_a.len() - 2] * row
                        + stride_a[stride_a.len() - 1] * i
                        + jump_a * j;

                    let index_b = stride_b[stride_b.len() - 2] * i
                        + stride_b[stride_b.len() - 1] * coll
                        + j * jump_b;

                    sum += data_a[index_a as usize] * data_b[index_b as usize];
                }
                output.push(sum);
            }
        }
    }

    output
}
