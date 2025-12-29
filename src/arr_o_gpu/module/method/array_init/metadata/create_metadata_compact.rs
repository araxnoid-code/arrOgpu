use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, BindingType, BufferUsages, ShaderStages, util::DeviceExt,
};

use crate::{
    ArrOgpuModule, ArrayMetadata,
    arr_o_gpu::module::method::array_init::metadata::metadata::MetadataCompound,
};

impl ArrOgpuModule {
    pub fn create_metadat_compound(
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
        };

        let wgpu = self.wgpu_init.read().unwrap();

        let metadata_buffer = wgpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Create Array Metadata, build/0.1.0.5"),
                contents: bytemuck::bytes_of(&metadata),
                usage: BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            wgpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Create Bind Group Layout For Array Metadata, build/0.1.0.5"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        count: None,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        visibility: ShaderStages::COMPUTE,
                    }],
                });

        let bind_group = wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Create Bind Group For Array Metadata, build/0.1.0.5"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: metadata_buffer.as_entire_binding(),
            }],
        });

        let metadata_compound = MetadataCompound {
            buffer: metadata_buffer,
            bind_group,
            bind_group_layout,
        };

        metadata_compound
    }
}
