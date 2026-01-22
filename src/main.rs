use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};

fn main() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        heap_size: arr_o_gpu::HeapSize::Item(4500000),
        ..Default::default()
    })
    .unwrap();

    let data = [1.; 2500];
    let array = module.array_from_vector(&data, &[50, 50]).unwrap();
    // let array = module.broadcast(&array, &[3, 2, 500]).unwrap();
    // println!("{}", array);

    let sum_axis = module.sum_axis_keep_dim(&array, &[0]).unwrap();
    println!("{}", sum_axis);
}

// println!("{:?}", &module.get_heap()[1000001..1000101]);
