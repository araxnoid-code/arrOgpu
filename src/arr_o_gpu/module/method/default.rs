use std::{
    sync::{Arc, RwLock},
    u32,
};

use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, BindingType, BufferUsages, ComputePipelineDescriptor,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
    ShaderStages,
    util::DeviceExt,
    wgt::{BufferDescriptor, PollType},
};

use crate::*;
// default
impl Default for ArrOgpuModule {
    fn default() -> Self {
        // create heap
        // let maximum = 5097152u32;
        // let maximum = 750_000u32;
        // let maximum = 268435456u32;
        // let maximum = 100_000u32;
        let maximum = 32_217_728u32;
        // let maximum = 750u32;

        //
        let wgpu = WgpuInit::init();

        let buffer_heap = wgpu.device.create_buffer(
            &(BufferDescriptor {
                label: Some("create heap"),
                mapped_at_creation: false,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                size: ((std::mem::size_of::<f32>() as u32) * maximum) as u64,
            }),
        );

        // group 0
        let binding_layout = wgpu.device.create_bind_group_layout(
            &(wgpu::BindGroupLayoutDescriptor {
                label: Some("create binding layout for heap"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    count: None,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                }],
            }),
        );

        let binding = wgpu.device.create_bind_group(
            &(wgpu::BindGroupDescriptor {
                label: Some("create binding layout for heap"),
                layout: &binding_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: buffer_heap.as_entire_binding(),
                }],
            }),
        );

        let shaders = wgpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("get shader"),
            source: ShaderSource::Wgsl(include_str!("./../../shader/shaders/init.wgsl").into()),
        });

        let pipeline_layout = wgpu.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("create pipeline layout for initialization"),
                bind_group_layouts: &[&binding_layout],
                push_constant_ranges: &[],
            }),
        );

        let pipeline = wgpu.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("create compute pipeline for initialization"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                layout: Some(&pipeline_layout),
                entry_point: Some("init"),
                module: &shaders,
            }),
        );

        let mut encoder = wgpu.device.create_command_encoder(
            &(wgpu::wgt::CommandEncoderDescriptor {
                label: Some("create encoder to push heap"),
            }),
        );

        {
            let mut bcp = encoder.begin_compute_pass(
                &(wgpu::ComputePassDescriptor {
                    label: Some("begin compute pass"),
                    timestamp_writes: None,
                }),
            );

            bcp.set_pipeline(&pipeline);
            bcp.set_bind_group(0, &binding, &[]);
            bcp.dispatch_workgroups(1, 1, 1);
        }

        wgpu.queue.submit(Some(encoder.finish()));
        wgpu.device.poll(PollType::Wait).unwrap();

        Self {
            allocator: Arc::new(RwLock::new(Allocator::init(maximum))),
            maximum: Arc::new(maximum),
            wgpu_init: Arc::new(RwLock::new(wgpu)),
            binding_compounds: Arc::new(RwLock::new(vec![BindGroupCompound {
                group: 0,
                binding_group_layouts: binding_layout,
                binding_groups: binding,
            }])),
            heap_buffer: Arc::new(buffer_heap),
        }
    }
}
