use arr_o_gpu::{
    AddOpt, ArangeArray, ArangeIteratorTrait, ArrOgpuModule, ArrOgpuModuleInit, FunctionExecuteOpt,
    WgpuLimits,
};

fn main() {
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
    println!("{:#?}", module.get_pipeline_cache().read());
}
