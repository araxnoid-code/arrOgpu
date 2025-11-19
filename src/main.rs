use std::time::SystemTime;

use arr_o_gpu::{ ArrOgpuModule, r };

fn main() {
    let module = ArrOgpuModule::default();

    let data = (0..3 * 3 * 2 * 6).map(|v| v as f32).collect::<Vec<f32>>();
    // let data = [999.0; 12];
    let array = module.array_from_vector(&data, &[3, 3, 2, 6]).unwrap();
    println!("{}", array);

    let array_a = module.slicing(&array, &[r(1..3), r(0..2), r(1..2), r(0..3)]).unwrap();

    println!("{}", array_a);
    println!();
    println!("{:?}", module.get_heap());
}
