use std::time::SystemTime;

use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    // let data = (0..16384).map(|x| x as f32).collect::<Vec<f32>>();
    // let array = module.array_from_vector(&data, &[32, 512, 1]).unwrap();
    // // println!("{}", array);

    // let tik = std::time::SystemTime
    //     ::now()
    //     .duration_since(SystemTime::UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    // let arr = module.broadcasting(&array, &[32, 512, 32]).unwrap();

    // let tok = std::time::SystemTime
    //     ::now()
    //     .duration_since(SystemTime::UNIX_EPOCH)
    //     .unwrap()
    //     .as_millis();

    // println!("{}", tok - tik)

    // // // // // //

    let data = (0..20).map(|x| x as f32).collect::<Vec<f32>>();

    let array = module.array_from_vector(&data, &[1, 4, 5]).unwrap();
    println!("{}", array);

    let arr = module.broadcasting(&array, &[4, 4, 5]).unwrap();
    println!("{}", arr);

    println!("{:?}", module.get_heap());
}
