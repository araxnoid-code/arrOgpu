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
        shape: [u32; 10],
        stride: [u32; 10],
        origin_stride: [u32; 10],
    ) -> MetadataCompound {
        let metadata = ArrayMetadata {
            pointer,
            len,
            dim,
            offset,
            shape,
            stride,
            origin_stride,
            padding: [0; 29],
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
