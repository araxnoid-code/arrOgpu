use arr_o_gpu::{
    AddOpt, ArangeArray, ArangeIteratorTrait, ArrOgpuModule, ArrOgpuModuleInit, FunctionExecuteOpt,
    MATMUL_2D_SHADERS_PATH, WgpuLimits,
};
use ndarray::Array2;

fn main() {
    let module = ArrOgpuModule::init(crate::ArrOgpuModuleInit {
        heap_size: arr_o_gpu::HeapSize::Item(1_000_000),
        function_execute_opt: FunctionExecuteOpt {
            matmul: arr_o_gpu::MatmulOpt { workgroup_size: 16 },
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();

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
    println!("{}", result.get_heap() == check_result);
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
