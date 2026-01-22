use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};

fn main() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        heap_size: arr_o_gpu::HeapSize::Item(4500000),
        ..Default::default()
    })
    .unwrap();

    let data = [1.; 1000];
    let array = module.array_from_vector(&data, &[2, 500]).unwrap();
    let array = module.broadcast(&array, &[3, 2, 500]).unwrap();
    // println!("{}", array);

    let sum_axis = module.sum_axis_unsafe(&array, &[1]).unwrap();
    println!("{}", module.sum(&sum_axis).unwrap());
}

// println!("{:?}", &module.get_heap()[1000001..1000101]);
