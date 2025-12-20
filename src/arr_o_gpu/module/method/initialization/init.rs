use std::sync::{ Arc, RwLock };

use wgpu::{
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayoutEntry,
    BufferUsages,
    ShaderStages,
    wgt::BufferDescriptor,
};

use crate::{
    Allocator,
    ArrOgpuErr,
    ArrOgpuModule,
    BindGroupCompound,
    WgpuModule,
    arr_o_gpu::module::method::initialization::module_init::ArrOgpuModuleInit,
};

impl ArrOgpuModule {
    pub fn init(arr_o_gpu_module_init: ArrOgpuModuleInit) -> Result<ArrOgpuModule, ArrOgpuErr> {
        let (maksimum, size) = arr_o_gpu_module_init.heap_size.get_size_of_32()?;
        let wgpu = WgpuModule::init(arr_o_gpu_module_init);

        let heap_buffer = wgpu.device.create_buffer(
            &(BufferDescriptor {
                label: Some("Create Heap For Init"),
                mapped_at_creation: false,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                size,
            })
        );

        let bind_group_layout = wgpu.device.create_bind_group_layout(
            &(wgpu::BindGroupLayoutDescriptor {
                label: Some("Crate Heap Bind Group Layout For Init"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        count: None,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        visibility: ShaderStages::COMPUTE,
                    },
                ],
            })
        );

        let bind_group = wgpu.device.create_bind_group(
            &(BindGroupDescriptor {
                label: Some("Create Heap Bind Group For Init"),
                layout: &bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: heap_buffer.as_entire_binding(),
                    },
                ],
            })
        );

        let compound = BindGroupCompound {
            group: 0,
            binding_group_layouts: bind_group_layout,
            binding_groups: bind_group,
        };

        Ok(Self {
            allocator: Arc::new(RwLock::new(Allocator::init(maksimum))),
            maximum: Arc::new(maksimum),
            wgpu_init: Arc::new(RwLock::new(wgpu)),
            heap_binding: Arc::new(compound),
            heap_buffer: Arc::new(heap_buffer),
        })
    }
}
