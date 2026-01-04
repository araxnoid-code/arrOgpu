use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutEntry, BufferUsages, ShaderStages,
    wgt::BufferDescriptor,
};

use crate::{
    Allocator, ArrOgpuErr, ArrOgpuModule, BindGroupCompound, WgpuModule,
    arr_o_gpu::module::method::initialization::module_init::ArrOgpuModuleInit,
};

impl ArrOgpuModule {
    pub fn init(arr_o_gpu_module_init: ArrOgpuModuleInit) -> Result<ArrOgpuModule, ArrOgpuErr> {
        let (maksimum, size) = arr_o_gpu_module_init.heap_size.get_size_of_32()?;
        let wgpu = WgpuModule::init(arr_o_gpu_module_init);

        // storage
        // // heap
        let heap_buffer = wgpu.device.create_buffer(
            &(BufferDescriptor {
                label: Some("Create Heap For Init"),
                mapped_at_creation: false,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                size,
            }),
        );

        // // execute_array_cache
        let size_cache = 768;
        let exececute_args_buffer = wgpu.device.create_buffer(&BufferDescriptor {
            label: Some("Create execute args cache For Init"),
            size: size_cache,
            usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        // // static_cache
        let size_cache = 32;
        let static_cache = wgpu.device.create_buffer(&BufferDescriptor {
            label: Some("Create Static Cache"),
            size: size_cache,
            usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        // bind group
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
                    BindGroupLayoutEntry {
                        binding: 1,
                        count: None,
                        visibility: ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        count: None,
                        visibility: ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                ],
            }),
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
                    BindGroupEntry {
                        binding: 1,
                        resource: exececute_args_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: static_cache.as_entire_binding(),
                    },
                ],
            }),
        );

        let buffer_compound = BindGroupCompound {
            binding_group_layouts: bind_group_layout,
            binding_groups: bind_group,
        };

        Ok(Self {
            allocator: Arc::new(RwLock::new(Allocator::init(maksimum))),
            maximum: Arc::new(maksimum),
            wgpu_init: Arc::new(RwLock::new(wgpu)),

            // Module Bind
            module_bind_group: Arc::new(buffer_compound),

            // cache
            // // pipeline
            pipeline_cache: Arc::new(RwLock::new(HashMap::new())),

            // Module Buffer
            // // Heap
            heap_buffer: Arc::new(heap_buffer),
            // // execute_array_cache
            execute_args: Arc::new(exececute_args_buffer),
            // // static
            static_cache: Arc::new(static_cache),
        })
    }
}
