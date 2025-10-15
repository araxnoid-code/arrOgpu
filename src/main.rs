use std::{
    cell::RefCell,
    collections::{ BTreeMap, VecDeque },
    ops::Range,
    rc::Rc,
    sync::{ Mutex, RwLock },
};

use arr_o_gpu::ArrOgpuModule;

fn main() {
    let module = ArrOgpuModule::default();

    println!("{:?} (1)", module.allocator_read().monanagement.range_space);
    {
        let _a = module.array_from_vector(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[1, 10]
        );
        // let _b = module.array_from_vector(&[11.0, 12.0, 13.0, 14.0], &[1, 4]);
        // println!("{:?} (2)", module.allocator_read().range_space());
        println!("{:?} (2)", module.allocator_read().monanagement.range_space);
    }
    // println!("{:?} (3)", module.allocator_read().range_space());
    println!("{:?} (3)", module.allocator_read().monanagement.range_space);
    println!("{:?}", module.get_heap());

    {
        let _a = module.array_from_vector(&[51.0, 52.0, 53.0, 54.0], &[1, 4]);
        println!("{:?} (4)", module.allocator_read().monanagement.range_space);

        println!("{:?}", module.get_heap());

        let _b = module.array_from_vector(&[101.0, 102.0, 103.0], &[1, 3]);

        println!("{:?} (4)", module.allocator_read().monanagement.range_space);

        println!("{:?}", module.get_heap());

        let _c = module.array_from_vector(&[201.0, 202.0, 203.0], &[1, 3]);

        println!("{:?} (4)", module.allocator_read().monanagement.range_space);
        // let _c = module.array_from_vector(&[101.0, 102.0, 103.0], &[1, 3]);
        // println!("{:?} (4)", module.allocator_read().monanagement.range_space);

        // println!("{:?} (4)", module.allocator_read().range_space());
        // let _b = module.array_from_vector(&[100.0, 101.0], &[1, 2]);
        // println!("{:?} (5)", module.allocator_read().range_space());
    }
    println!("{:?} (5)", module.allocator_read().monanagement.range_space);

    println!("{:?}", module.get_heap());
}
