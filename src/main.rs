use std::time::SystemTime;

use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    let array = module
        .array_from_vector(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[2, 1, 4])
        .unwrap();
    println!("{}", array);

    let arr = module.broadcasting(&array, &[2, 2, 4]).unwrap();
    println!("{}", arr);

    println!("{:?}", module.get_heap())
}
