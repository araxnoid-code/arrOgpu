use std::sync::Arc;

use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BufferBindingType, BufferUsages, ComputePipelineDescriptor, Device,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderStages,
    util::{BufferInitDescriptor, DeviceExt},
    wgt::CommandEncoderDescriptor,
};

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, GpuArray,
    arr_o_gpu::module::method::function::operation::element_wise::skalar_operation::MetaDataOption,
    get_stride_from_shape,
};

impl ArrOgpuModule {
    pub(crate) fn sub_skalar<A>(
        &self,
        array: &A,
        meta_data_option: MetaDataOption,
    ) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute,
    {
        let wgpu = self.wgpu_init.read().unwrap();

        // output metadata
        let shape = array.shape();
        let len = array.len();
        let stride = get_stride_from_shape(&shape);
        let allocate = self.allocator_write().pointer_input(len);

        // binding
        // // heap
        let heap_bind = &self.heap_binding;

        // // array
        let array_bind = array.binding();

        // // array b
        let array_b_bind = match &meta_data_option {
            MetaDataOption::Array(arr) => arr.binding(),
            MetaDataOption::Skalar(scalar) => &scalar_binding(&wgpu.device, scalar),
        };

        // // output
        let out_bind =
            self.create_metadata_binding(&[allocate.1, allocate.2], &shape, &stride, &stride, &0);

        // pipeline
        let shader = wgpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Create Shader Module FOr Add Scalar"),
            source: match meta_data_option {
                MetaDataOption::Array(_) => {
                    wgpu::ShaderSource::Wgsl(include_str!("./wgsl/sub_array_skalar.wgsl").into())
                }
                MetaDataOption::Skalar(_) => {
                    wgpu::ShaderSource::Wgsl(include_str!("./wgsl/sub_skalar.wgsl").into())
                }
            },
        });

        let pipeline_layout = wgpu.device.create_pipeline_layout(
            &(PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Add Scalar"),
                immediate_size:0,
                bind_group_layouts: &[
                    &heap_bind.binding_group_layouts,
                    &array_bind.0,
                    &array_b_bind.0,
                    &out_bind.0,
                ],
            }),
        );

        let pipeline = wgpu.device.create_compute_pipeline(
            &(ComputePipelineDescriptor {
                label: Some("Create Pipeline For Add Scalar"),
                cache: None,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions::default(),
                layout: Some(&pipeline_layout),
                module: &shader,
            }),
        );

        let mut encoder = wgpu.device.create_command_encoder(
            &(CommandEncoderDescriptor {
                label: Some("Create Encoder For Add Scalar"),
            }),
        );

        {
            let mut bcp = encoder.begin_compute_pass(
                &(wgpu::ComputePassDescriptor {
                    label: Some("Create Begin Compute Pass For Add Scalar"),
                    timestamp_writes: None,
                }),
            );

            bcp.set_pipeline(&pipeline);

            // 0
            bcp.set_bind_group(0, Some(&heap_bind.binding_groups), &[]);
            // 1
            bcp.set_bind_group(1, Some(&array_bind.1), &[]);
            // 2
            bcp.set_bind_group(2, Some(&array_b_bind.1), &[]);
            // 3
            bcp.set_bind_group(3, Some(&out_bind.1), &[]);

            let x = (len + 256 - 1) / 256;
            bcp.dispatch_workgroups(x, 1, 1);
        }

        wgpu.queue.submit(Some(encoder.finish()));

        // if let Err(poll_err) = wgpu.device.poll(wgpu::wgt::PollType::Wait) {
        //     let error = "Add Error, Error While Poll".to_string();
        //     return Err(ArrOgpuErr::Poll(error, poll_err));
        // }

        let array = GpuArray {
            module: Arc::new(self.clone()),
            pointer: (allocate.1, allocate.2),
            shape: shape.clone(),
            stride,
            space_type: allocate.0,
            length: len as usize,
            binding: out_bind,
        };

        Ok(array)
    }
}

fn scalar_binding(device: &Device, scalar: &f32) -> (BindGroupLayout, wgpu::BindGroup) {
    // buffer
    let scalar_buffer = device.create_buffer_init(
        &(BufferInitDescriptor {
            label: Some("Create Buffer Scalar For Add Scalar"),
            usage: BufferUsages::UNIFORM,
            contents: bytemuck::bytes_of(scalar),
        }),
    );

    // bind_group_layout
    let bind_group_layout = device.create_bind_group_layout(
        &(BindGroupLayoutDescriptor {
            label: Some("Create Bind Group Layout For Add Scalar"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                visibility: ShaderStages::COMPUTE,
            }],
        }),
    );

    // bind_group
    let bind_group = device.create_bind_group(
        &(BindGroupDescriptor {
            label: Some("Create Bind Group Layout For Add Scalar"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: scalar_buffer.as_entire_binding(),
            }],
        }),
    );

    (bind_group_layout, bind_group)
}
