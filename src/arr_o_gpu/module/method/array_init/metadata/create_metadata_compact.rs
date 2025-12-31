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
        shape: [u32; 12],
        stride: [u32; 12],
        origin_stride: [u32; 12],
    ) -> MetadataCompound {
        let metadata = ArrayMetadata {
            pointer,
            len,
            offset,
            dim,
            padding_0: 0,
            padding_1: 0,
            padding_2: 0,
            shape,
            stride,
            origin_stride,
            padding_3: [0; 20],
        };

        let wgpu = self.wgpu_init.read().unwrap();

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
