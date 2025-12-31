use arr_o_gpu::{ArangeArray, ArangeIteratorTrait, ArrOgpuModule, ArrayCompute, HeapSize, r};

fn main() {
    let module = ArrOgpuModule::init(arr_o_gpu::ArrOgpuModuleInit {
        heap_size: HeapSize::Item(100000),
        wgpu: arr_o_gpu::WgpuInit::ManualInit(arr_o_gpu::ManualInit {
            power: arr_o_gpu::Power::LowPower,
            memory: arr_o_gpu::Memory::MemoryUsage,
        }),
    })
    .unwrap();

    // let array_a_view = array_a.broadcast(&[3, 3, 4]).unwrap();
    // let array_a_view = array_a_view.index(&[1]).unwrap();
    // let array_a_view = array_a_view.slicing(&[r(1..), r(1..)]).unwrap();
    // let array_a_view = array_a_view.index(&[0]).unwrap();

    // let array_b_view = array_b.slicing(&[r(1..3)]).unwrap();
    // let array_b_view = array_b_view.broadcast(&[3, 2, 3, 3]).unwrap();
    // let array_b_view = array_b_view.slicing(&[r(1..), r(..1)]).unwrap();
    // let array_b_view = array_b_view.index(&[1, 0, 2]).unwrap();

    // println!("{:?}", &module.get_heap()[25..50]);
    // println!("{}", array_a_view.contiguous_metadata().unwrap());
    // println!("{}", array_b_view.contiguous_metadata().unwrap());

    // let array_contiguous_a = array_a_view.contiguous_metadata().unwrap().get_heap();

    // let array_contiguous_b = array_b_view.contiguous_metadata().unwrap().get_heap();

    // let dot = array_contiguous_a
    //     .iter()
    //     .zip(array_contiguous_b.iter())
    //     .map(|(&a, &b)| a * b)
    //     .sum::<f32>();
    // println!("{}", dot);
}
