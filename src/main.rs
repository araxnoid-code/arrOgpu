use std::time;

use arr_o_gpu::{ ArrOgpuModule, GpuArray };

fn main() {
    let module = ArrOgpuModule::default();

    // let array_a = module
    //     .array_from_vector(&(0..1048576).map(|v| v as f32).collect::<Vec<f32>>(), &[1024, 1024])
    //     .unwrap();

    // let array_b = module
    //     .array_from_vector(&(0..1048576).map(|v| v as f32).collect::<Vec<f32>>(), &[1024, 1024])
    //     .unwrap();

    let array_a = module
        .array_from_vector(&(1..16).map(|v| v as f32).collect::<Vec<f32>>(), &[5, 3])
        .unwrap();

    let array_b = module
        .array_from_vector(&(1..16).map(|v| v as f32).collect::<Vec<f32>>(), &[5, 3])
        .unwrap();

    // let tick = std::time::SystemTime::now().duration_since(time::UNIX_EPOCH).unwrap().as_millis();
    let out = module.div(array_a, array_b).unwrap();
    // let tock = std::time::SystemTime::now().duration_since(time::UNIX_EPOCH).unwrap().as_millis();
    // println!("{}", tock - tick);
    println!("{out}");
    println!("{:?}", module.get_heap());
}
