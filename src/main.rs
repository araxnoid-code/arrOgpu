use std::{ sync::{ Arc, Mutex }, thread::spawn };

use arr_o_gpu::{ ArrOgpuModule, FlatteTrait };

fn main() {
    let mut module = ArrOgpuModule::default();

    // let array_a = module.array_from_vector(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5]);
    // let array_b = module.array_from_vector(
    //     &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    //     &[1, 10]
    // );
    // println!("{:?}", module.get_heap());
    // println!("{:?}", array_b.get_heap());
}
