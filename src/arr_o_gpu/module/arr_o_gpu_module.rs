use std::{ ops::Range, sync::{ Arc, RwLock } };

use wgpu::{
    util::DeviceExt,
    BindGroupEntry,
    BindGroupLayoutEntry,
    BindingType,
    BufferUsages,
    ComputePipelineDescriptor,
    PipelineCompilationOptions,
    PipelineLayoutDescriptor,
    ShaderModule,
    ShaderModuleDescriptor,
    ShaderSource,
    ShaderStages,
};

use crate::arr_o_gpu::WgpuInit;

pub struct ArrOgpuModule {
    free: Vec<Range<usize>>,
    max_size: usize,
    wgpu_init: Arc<RwLock<WgpuInit>>,
    main_shader: ShaderModule,
}

impl Default for ArrOgpuModule {
    fn default() -> Self {
        // create heap
        let max_size = 1_000_000;
        let wgpu = WgpuInit::init();

        let heap: Vec<f32> = vec![0.0; max_size];
        let buffer_heap = wgpu.device.create_buffer_init(
            &(wgpu::util::BufferInitDescriptor {
                label: Some("create heap"),
                contents: bytemuck::cast_slice(&heap),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            })
        );

        let binding_layout = wgpu.device.create_bind_group_layout(
            &(wgpu::BindGroupLayoutDescriptor {
                label: Some("create binding layout for heap"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        count: None,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                ],
            })
        );

        let binding = wgpu.device.create_bind_group(
            &(wgpu::BindGroupDescriptor {
                label: Some("create binding layout for heap"),
                layout: &binding_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: buffer_heap.as_entire_binding(),
                    },
                ],
            })
        );

        let shaders = wgpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("get shader"),
            source: ShaderSource::Wgsl(include_str!("./../shaders/main.wgsl").into()),
        });

        let pipeline_layout = wgpu.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("create pipeline layout for initialization"),
                bind_group_layouts: &[&binding_layout],
                push_constant_ranges: &[],
            })
        );

        let pipeline = wgpu.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("create compute pipeline for initialization"),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                layout: Some(&pipeline_layout),
                entry_point: Some("init"),
                module: &shaders,
            })
        );

        let mut encoder = wgpu.device.create_command_encoder(
            &(wgpu::wgt::CommandEncoderDescriptor {
                label: Some("create encoder to push heap"),
            })
        );

        {
            let mut bcp = encoder.begin_compute_pass(
                &(wgpu::ComputePassDescriptor {
                    label: Some("begin compute pass"),
                    timestamp_writes: None,
                })
            );

            bcp.set_pipeline(&pipeline);
            bcp.set_bind_group(0, &binding, &[]);
            bcp.dispatch_workgroups(1, 1, 1);
        }

        wgpu.queue.submit(Some(encoder.finish()));

        Self {
            free: Vec::new(),
            max_size,
            wgpu_init: Arc::new(RwLock::new(wgpu)),
            main_shader: shaders,
        }
    }
}
