use std::{time::UNIX_EPOCH, vec};

use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, ArrayCompute, r};

fn main() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        heap_size: arr_o_gpu::HeapSize::Item(5_000_000),
        wgpu: arr_o_gpu::WgpuInit::ManualInit(arr_o_gpu::ManualInit {
            power: arr_o_gpu::Power::HighPerformance,
            memory: arr_o_gpu::Memory::Performance,
        }),
    })
    .unwrap();

    let times = 120;
    let mut speed = 0;
    for _ in 0..times {
        let data = vec![1.; 2_000_000];
        let array = module.array_from_vector(&data, &[2_000_000]).unwrap();

        let tick = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        module.add(&array, &10.).unwrap();

        let tock = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        speed += tock - tick;
    }

    println!(" rata rata metode baru {} ms", speed as f64 / times as f64);

    let times = 120;
    let mut speed_cache = 0;
    for _ in 0..times {
        let data = vec![1.; 2_000_000];
        let array = module.array_from_vector(&data, &[2_000_000]).unwrap();

        let tick = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        module.sub(&array, &10.).unwrap();

        let tock = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        speed_cache += tock - tick;
    }

    println!(
        " rata rata metode lama {} ms",
        speed_cache as f64 / times as f64
    );

    let peningkatan = (speed_cache as f64 - speed as f64) / speed_cache as f64;
    println!(" persentase peningkatan {}%", peningkatan * 100.0);
}
