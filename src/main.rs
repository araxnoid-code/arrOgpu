use std::time::UNIX_EPOCH;

use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule};
use ndarray::Array2;

fn main() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        heap_size: arr_o_gpu::HeapSize::Item(3000000),
        wgpu: arr_o_gpu::WgpuInit::ManualInit(arr_o_gpu::ManualInit {
            power: arr_o_gpu::Power::HighPerformance,
            memory: arr_o_gpu::Memory::Performance,
        }),
    })
    .unwrap();

    let new = {
        let array = ArangeArray::arange(0..1048576)
            .to_GpuArray_with_shape(&[1024, 1024], &module)
            .unwrap();

        let tick = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        let matmul = module.matmul_2d_metadata(&array, &array).unwrap();

        let tock = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        println!("{} ms", tock - tick);
        tock - tick
    };

    let old = {
        let data = (0..1048576).map(|v| v as f32).collect::<Vec<f32>>();
        let array = Array2::from_shape_vec([1024, 1024], data).unwrap();

        let tick = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        let matmul = array.dot(&array);

        let tock = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        println!("{} ms", tock - tick);
        tock - tick
    };

    // println!("peningkatan");
    // let persen = ((old as f64 - new as f64) / old as f64) * 100.;
    // println!("{}", persen)
}
