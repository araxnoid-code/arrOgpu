use arr_o_gpu::{
    AddOpt, ArangeArray, ArangeIteratorTrait, ArrOgpuModule, ArrOgpuModuleInit, FunctionExecuteOpt,
    WgpuLimits,
};

fn main() {
    let module = ArrOgpuModule::init(ArrOgpuModuleInit {
        heap_size: arr_o_gpu::HeapSize::Item(100_000),
        limits: WgpuLimits {
            max_compute_workgroup_size_x: 512,
            max_compute_invocations_per_workgroup: 512,
            ..Default::default()
        },
        function_execute_opt: FunctionExecuteOpt {
            add: AddOpt {
                compute_workgroup_size_x: 256,
            },
        },
        ..Default::default()
    })
    .unwrap();

    let array_a = ArangeArray::arange(0..1024)
        .to_GpuArray_with_shape(&[32, 32], &module)
        .unwrap();

    let array_b = ArangeArray::arange(1024..2048)
        .to_GpuArray_with_shape(&[32, 32], &module)
        .unwrap();

    let sum = module.add(&array_a, &array_b).unwrap();
}
