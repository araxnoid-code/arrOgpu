use arr_o_gpu::{ ArrOgpuModule, FlatteTrait };

fn main() {
    let mut module = ArrOgpuModule::default();

    // println!("{:?}", module.allocator.last_space);
    let array_a = module.array_from_vector(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 5]);
    // println!("{:?}", module.allocator.last_space);
    let array_b = module.array_from_vector(
        &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        &[1, 10]
    );
    // println!("{:?}", module.allocator.last_space);
    // println!("{:?}", array_a.get_heap_pointer())
}
