use std::{
    ops::{Range, RangeFull},
    sync::mpsc,
    time::UNIX_EPOCH,
};

use arr_o_gpu::{ArangeArray, ArrOgpuModule, FlatteTrait};

fn main() {
    let data = (0..2097152)
        .into_iter()
        .map(|v| v as f32)
        .collect::<Vec<f32>>();
    // let array_a = Array2::from_shape_vec([1024, 2048], data.clone()).unwrap();
    // let array_b = Array2::from_shape_vec([2048, 1024], data.clone()).unwrap();

    // let tick = std::time::SystemTime::now()
    //     .duration_since(UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();
    // let to_vec_numpy = array_a.dot(&array_b).flatten().to_vec();
    // let tock = std::time::SystemTime::now()
    //     .duration_since(UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    // println!("{:?}", tock - tick)

    // println!("==============");

    let module = ArrOgpuModule::default();

    let array_a = module.array_from_vector(&data, &[1024, 2048]).unwrap();

    let array_b = module.array_from_vector(&data, &[2048, 1024]).unwrap();

    let tick = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let to_vec_my = module.matmul_nd(&array_a, &array_b).unwrap().get_heap();
    let tock = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();

    println!("{:?}", tock - tick)
}
