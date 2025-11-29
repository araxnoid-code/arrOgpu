use std::{ ops::{ Range, RangeFull }, sync::mpsc, time::UNIX_EPOCH };

use arr_o_gpu::{ ArangeArray, ArrOgpuModule, FlatteTrait };
use ndarray::{ Array1, Array2 };

fn main() {
    let len = 32 * 33;
    let data = (0..len)
        .into_iter()
        .map(|v| v as f32)
        .collect::<Vec<f32>>();
    let array_a = Array2::from_shape_vec([32, 33], data.clone()).unwrap();
    let array_b = Array2::from_shape_vec([33, 32], data.clone()).unwrap();

    // let tick = std::time::SystemTime::now()
    //     .duration_since(UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();
    let to_vec_numpy = array_a.dot(&array_b);
    // let tock = std::time::SystemTime::now()
    //     .duration_since(UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    // println!("{:?}", tock - tick)

    // println!("==============");

    let module = ArrOgpuModule::default();

    let array_a = module.array_from_vector(&data, &[32, 33]).unwrap();
    let array_b = module.array_from_vector(&data, &[33, 32]).unwrap();

    // let array_b = module.array_from_vector(&data, &[len as u32]).unwrap();

    // let tick = std::time::SystemTime::now()
    //     .duration_since(UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();
    let to_vec_my = module.matmul_2d(&array_a, &array_b).unwrap();
    // let tock = std::time::SystemTime::now()
    //     .duration_since(UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    println!("my:\t{}\nother:\t{}", to_vec_my, to_vec_numpy);
    println!("{}", to_vec_my.get_heap() == to_vec_numpy.flatten().to_vec());
}
