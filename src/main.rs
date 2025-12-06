use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    let array = module.array_from_vector(&[0.0, 1.0, 2.0, 3.0], &[2, 2]).unwrap();
    let indexing = module.index_view(&array, &[0]).unwrap();
}
