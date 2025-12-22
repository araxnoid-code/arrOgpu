use std::time::SystemTime;

use arr_o_gpu::{
    ArangeArray,
    ArangeIteratorTrait,
    ArrOgpuModule,
    ArrOgpuModuleInit,
    HeapSize,
    ManualInit,
    Memory,
    Power,
    WgpuInit,
};
// use ndarray::{ Array1, Array2, Array3 };

fn main() {
    // let shape = [2048, 2048];
    // let data = vec![10.0; 4_194_304];
    let module = ArrOgpuModule::init(ArrOgpuModuleInit {
        heap_size: HeapSize::Item(262_756),
        wgpu: WgpuInit::ManualInit(ManualInit {
            memory: Memory::Performance,
            power: Power::HighPerformance,
        }),
    }).unwrap();

    let data = vec![10.; 61952];
    let array = module.array_from_vector(&data, &[61952]).unwrap();

    let out = module.sum(&array).unwrap();

    println!("{}", out);
    println!("{:?}", &module.get_heap()[61953..62500]);
    println!("{}", data.iter().sum::<f32>())

    // let gpu_array = module.array_from_vector(&data, &shape).unwrap();
    // let gpu_array_2 = module.array_from_vector(&vec![10.0; 100_000], &[100_000]).unwrap();

    // let tick = std::time::SystemTime
    //     ::now()
    //     .duration_since(SystemTime::UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    // let sum = module.sum(&gpu_array).unwrap();
    // let add = module.add(&gpu_array_2, &sum).unwrap();

    // let tock = std::time::SystemTime
    //     ::now()
    //     .duration_since(SystemTime::UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    // println!("{}", tock - tick);

    // let cpu_array = Array2::from_shape_vec(shape, data).unwrap();
    // // let cpu_array
    // let cpu_skalar = Array1::from_shape_vec([100_000], vec![10.0; 100_000]).unwrap();

    // let tick = std::time::SystemTime
    //     ::now()
    //     .duration_since(SystemTime::UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    // let sum = cpu_array.sum();
    // let add = cpu_skalar + sum;

    // let tock = std::time::SystemTime
    //     ::now()
    //     .duration_since(SystemTime::UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    // println!("{}", tock - tick)
}
