use arr_o_gpu::{ ArangeArray, ArangeIteratorTrait, ArrOgpuModule };

fn main() {
    let module = ArrOgpuModule::default();

    let array_a = ArangeArray::arange(0..6)
        .to_GpuArray_with_shape(&[2, 3], &module)
        .unwrap();

    let array_b = ArangeArray::arange(0..6)
        .to_GpuArray_with_shape(&[2, 3], &module)
        .unwrap();

    module.add_view(&array_a, &array_b);
    module.sub_view(&array_a, &array_b);
    module.mul_view(&array_a, &array_b);
    module.div_view(&array_a, &array_b);
}
