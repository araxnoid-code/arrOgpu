use wgpu::{
    util::DeviceExt,
    wgt::{ BufferDescriptor, CommandEncoderDescriptor, PollType },
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BindingType,
    BufferBindingType,
    BufferUsages,
    ComputePassDescriptor,
    ComputePipelineDescriptor,
    MapMode,
    PipelineCompilationOptions,
    PipelineLayoutDescriptor,
    ShaderModuleDescriptor,
    ShaderSource,
    ShaderStages,
};

use crate::ArrOgpuModule;

impl ArrOgpuModule {
    pub fn get_heap(&self) {
        let wgpu_init = &self.wgpu_init;
        let size = (std::mem::size_of::<f32>() * (self.maximum as usize)) as u64;

        let copy_buffer = wgpu_init.device.create_buffer(
            &(BufferDescriptor {
                label: Some(""),
                mapped_at_creation: false,
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            })
        );

        let result_buffer = wgpu_init.device.create_buffer(
            &(BufferDescriptor {
                label: Some("create result buffer for get_heap"),
                size,
                mapped_at_creation: false,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            })
        );

        // // group 1
        // let binding_layout = wgpu_init.device.create_bind_group_layout(
        //     &(BindGroupLayoutDescriptor {
        //         label: Some("create bind group layout for get_heap"),
        //         entries: &[
        //             BindGroupLayoutEntry {
        //                 binding: 0,
        //                 count: None,
        //                 visibility: ShaderStages::COMPUTE,
        //                 ty: BindingType::Buffer {
        //                     ty: BufferBindingType::Storage { read_only: false },
        //                     has_dynamic_offset: false,
        //                     min_binding_size: None,
        //                 },
        //             },
        //         ],
        //     })
        // );

        // let binding = wgpu_init.device.create_bind_group(
        //     &(BindGroupDescriptor {
        //         label: Some("create bind group for get_heap"),
        //         layout: &binding_layout,
        //         entries: &[
        //             BindGroupEntry {
        //                 binding: 0,
        //                 resource: copy_buffer.as_entire_binding(),
        //             },
        //         ],
        //     })
        // );

        // let pipeline_layout = wgpu_init.device.create_pipeline_layout(
        //     &(PipelineLayoutDescriptor {
        //         label: Some("create pipeline layout for get_heap"),
        //         bind_group_layouts: &[
        //             &self.binding_compounds[0].binding_group_layouts,
        //             &binding_layout,
        //         ],
        //         push_constant_ranges: &[],
        //     })
        // );

        // let shaders = wgpu_init.device.create_shader_module(ShaderModuleDescriptor {
        //     label: Some("create shaders module 'get_heap.wgsl'"),
        //     source: ShaderSource::Wgsl(include_str!("./../../shader/shaders/get_heap.wgsl").into()),
        // });

        // let pipeline = wgpu_init.device.create_compute_pipeline(
        //     &(ComputePipelineDescriptor {
        //         label: Some("create pipeline for get_heap"),
        //         cache: None,
        //         compilation_options: PipelineCompilationOptions::default(),
        //         entry_point: Some("main"),
        //         layout: Some(&pipeline_layout),
        //         module: &shaders,
        //     })
        // );

        let mut encoder = wgpu_init.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("create encoder for get_heap"),
            })
        );

        // {
        //     let mut bcp = encoder.begin_compute_pass(
        //         &(ComputePassDescriptor {
        //             label: Some("begin compute pass for get_heap"),
        //             timestamp_writes: None,
        //         })
        //     );

        //     bcp.set_pipeline(&pipeline);
        //     // group 0
        //     bcp.set_bind_group(0, &self.binding_compounds[0].binding_groups, &[]);
        //     // group 2
        //     bcp.set_bind_group(1, &binding, &[]);
        //     bcp.dispatch_workgroups(1, 1, 1);
        // }

        encoder.copy_buffer_to_buffer(&self.heap_buffer, 0, &result_buffer, 0, size);
        wgpu_init.queue.submit(Some(encoder.finish()));

        let buffer_slice = result_buffer.slice(..);
        buffer_slice.map_async(MapMode::Read, |e| e.unwrap());
        wgpu_init.device.poll(PollType::Wait).unwrap();

        let data = buffer_slice.get_mapped_range();
        let heap: Vec<f32> = bytemuck::cast_slice(&data).into();
        drop(data);
        result_buffer.unmap();

        println!("{:?}", heap)
    }
}
