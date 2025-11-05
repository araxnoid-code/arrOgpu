use std::time::SystemTime;

use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    let data = (0..12582912).map(|a| a as f32).collect::<Vec<f32>>();
    let array_a = module.array_from_vector(&data, &[48, 16, 16384]).unwrap();
    // println!("{}", array_a);

    let data = (0..12582912).map(|a| a as f32).collect::<Vec<f32>>();
    let array_b = module.array_from_vector(&data, &[48, 16384, 16]).unwrap();
    // println!("{}", array_b);

    let tik = std::time::SystemTime
        ::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis();

    let out = module.matmul_nd(&array_a, &array_b).unwrap();

    // println!("{}", out);

    let tok = std::time::SystemTime
        ::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis();

    println!("{}", tok - tik);
}
