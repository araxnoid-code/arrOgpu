use std::vec;

use bytemuck::{Pod, Zeroable};
use wgpu::BufferUsages;

use crate::{ArrOgpuErr, ArrOgpuModule, ArrayCompute};

impl ArrOgpuModule {
    pub fn dot_product_optimize<A, B>(&self, array_a: &A, array_b: &B) -> Result<(), ArrOgpuErr>
    where
        A: ArrayCompute,
        B: ArrayCompute,
    {
        let shape_a = array_a.shape();
        let shape_b = array_b.shape();
        if shape_a.len() != 1 && shape_b.len() != 1 {
            let err = format!(
                "Dot Product Error, Array A with shape {:?} and Array B With Shape {:?} Can't Be Operated",
                shape_a, shape_b
            );

            return Err(ArrOgpuErr::DotProduct(err));
        } else if array_a.len() != array_b.len() {
            let err = format!(
                "Dot Product Error, Array A And Array B Do Not Have The Same Length So They Cannot Be Operated On",
            );

            return Err(ArrOgpuErr::DotProduct(err));
        }

        let wgpu = self.wgpu_init().read().unwrap();

        // output metadata
        let len = 1;
        let shape = vec![1];
        let stride = vec![1];
        let allocate = self.allocator_write().pointer_input(len);

        // bind
        let heap_bind = self.heap_binding();
        let array_a_bind = array_a.binding();
        let array_b_bind = array_b.binding();
        let out_bind =
            self.create_metadata_binding(&[allocate.1, allocate.2], &shape, &stride, &stride, &0);

        // reduction
        // // reduction_counter
        let reduction_counter = create_reduction_counter();
        let window_reduction_counter = wgpu.device.create_buffer(&wgpu::wgt::BufferDescriptor {
            label: Some("Create Window Reduction Counter For Dot Product"),
            size: 256 * 2,
            usage: BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        // // reduction_len
        let reduction_len = create_reduction_len(array_a.len());

        let window_reduction_len = wgpu.device.create_buffer(&wgpu::wgt::BufferDescriptor {
            label: Some("Create Reduction For Dot Product"),
            size: (reduction_len.len() * 256) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // // reduction
        let out_len = (array_a.len() + 511) / 512;
        let reduction_buffer = wgpu.device.create_buffer(&wgpu::wgt::BufferDescriptor {
            label: Some("Create Reduction For Dot Product"),
            size: (out_len * 4) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Ok(())
    }
}

fn create_reduction_counter() -> [Counter; 2] {
    [
        Counter {
            value: 0,
            padding: [0; 63],
        },
        Counter {
            value: 1,
            padding: [0; 63],
        },
    ]
}

fn create_reduction_len(len: u32) -> Vec<Counter> {
    let mut reduction_len = vec![];
    let mut len = len;
    loop {
        let out_len = (len + 511) / 512;

        if out_len == 1 {
            reduction_len.push(Counter {
                value: 1,
                padding: [0; 63],
            });
            break;
        } else {
            len = out_len;
            reduction_len.push(Counter {
                value: len,
                padding: [0; 63],
            });
        }
    }

    reduction_len
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct Counter {
    value: u32,
    padding: [u32; 63],
}
