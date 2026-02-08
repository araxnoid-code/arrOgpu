use wgpu::{BufferUsages, util::DeviceExt};

use crate::{
    ArrOgpuModule, ArrayMetadata,
    arr_o_gpu::module::method::array_init::metadata::metadata::MetadataCompound,
};

impl ArrOgpuModule {
    pub fn create_metadata_compound(
        &self,
        pointer: [u32; 2],
        len: u32,
        dim: u32,
        offset: u32,
        shape: [u32; 8],
        stride: [u32; 8],
        o_stride: [u32; 8],
    ) -> MetadataCompound {
        let shape: [u32; 8] = shape[..8].try_into().unwrap();
        let stride: [u32; 8] = stride[..8].try_into().unwrap();
        let o_stride = o_stride[..8].try_into().unwrap();

        let metadata = ArrayMetadata {
            pointer,
            len,

            offset,
            //

            //
            dim,
            padding_0: 0,
            padding_1: 0,
            padding_2: 0,
            //
            //
            m_n_shape: get_magic_number(&shape, dim),
            m_n_o_stride: get_magic_number(&stride, dim),
            //
            //
            shape,
            stride,
            o_stride,
            padding_3: [0; 16],
        };

        let wgpu = self.wgpu_module.read().unwrap();

        let metadata_buffer = wgpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Create Array Metadata, build/0.1.0.5"),
                contents: bytemuck::bytes_of(&metadata),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
            });

        let metadata_compound = MetadataCompound {
            buffer: metadata_buffer,
        };

        metadata_compound
    }
}

fn get_magic_number(array: &[u32; 8], dim: u32) -> [u32; 8] {
    let bit32: f64 = (1u64 << 32) as f64;
    (0..8)
        .map(|i| {
            if i < dim {
                (bit32 / array[i as usize] as f64).ceil() as u32
            } else {
                0
            }
        })
        .collect::<Vec<u32>>()
        .try_into()
        .unwrap()
}
