use arr_o_gpu::{
    AddOpt, ArangeArray, ArangeIteratorTrait, ArrOgpuModule, ArrOgpuModuleInit, FunctionExecuteOpt,
    WgpuLimits,
};

fn main() {
    let module = ArrOgpuModule::init(ArrOgpuModuleInit {
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
    println!("{:#?}", module.get_device_limit());
}
