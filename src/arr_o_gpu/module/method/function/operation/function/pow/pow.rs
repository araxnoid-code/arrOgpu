use std::sync::Arc;

use wgpu::{
    BindGroupEntry, BindGroupLayoutEntry, BindingType, BufferUsages, Device,
    PipelineCompilationOptions, ShaderStages, util::DeviceExt,
};

use crate::{
    ArrOgpuErr, ArrOgpuModule, ArrayCompute, CheckArrayType, GpuArray,
    arr_o_gpu::module::method::function::operation::{
        element_wise::MetaDataOption, function::pow::AblePowType,
    },
    get_stride_from_shape,
};

impl ArrOgpuModule {
    pub fn pow<'a, A, P>(&self, array: &'a A, pow: P)
    where
        A: ArrayCompute + CheckArrayType<'a>,
        P: AblePowType,
    {
        match array.check() {
            crate::ArrayType::Contiguous(arr) => self.pow_contiguous(arr, pow),
            crate::ArrayType::View(arr) => panic!(""),
        };
        // out metadata
        // let len = array.len();
        // let shape = array.shape();
        // let stride = get_stride_from_shape(shape);
        // let allocate = self.allocator.write().unwrap().pointer_input(len);

        // bind
        // let heap_bind = self.heap_binding();
        // let array_bind = array.binding();
        // let out_bind =
        //     self.array_data_binding(&[allocate.1, allocate.2], shape, &stride, &stride, &0);
    }
}

impl ArrOgpuModule {
    pub(super) fn pow_contiguous<A, P>(&self, array: &A, pow: P) -> Result<GpuArray, ArrOgpuErr>
    where
        A: ArrayCompute + ?Sized,
        P: AblePowType,
    {
        let wgpu = self.wgpu_init().read().unwrap();

        // out metadata
        let len = array.len();
        let shape = array.shape();
        let stride = get_stride_from_shape(shape);
        let allocate = self.allocator.write().unwrap().pointer_input(len);

        // bind
        let heap_bind = self.heap_binding();
        let array_bind = array.binding();
        let out_bind =
            self.array_data_binding(&[allocate.1, allocate.2], shape, &stride, &stride, &0);
        let scalar_bind = match pow.get() {
            MetaDataOption::Skalar(scalar) => &self.scalar_bind(&wgpu.device, scalar),
            MetaDataOption::Array(arr) => arr.binding(),
        };

        // pipeline
        let pipeline_layout = wgpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Create Pipeline Layout For Pow"),
                bind_group_layouts: &[
                    &heap_bind.binding_group_layouts,
                    &array_bind.0,
                    &out_bind.0,
                    &scalar_bind.0,
                ],
                push_constant_ranges: &[],
            });

        let shaders = match pow.get() {
            MetaDataOption::Array(_) => {
                wgpu.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Create Shader Module For Pow"),
                        source: wgpu::ShaderSource::Wgsl(
                            include_str!("./wgsl/pow_contiguous/pow_scalar.wgsl").into(),
                        ),
                    })
            }
            MetaDataOption::Skalar(_) => {
                wgpu.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Create Shader Module For Pow"),
                        source: wgpu::ShaderSource::Wgsl(
                            include_str!("./wgsl/pow_view/pow_scalar.wgsl").into(),
                        ),
                    })
            }
        };
        let pipeline = wgpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Create Pipeline For Pow"),
                layout: Some(&pipeline_layout),
                module: &shaders,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });

        let mut encoder =
            wgpu.device
                .create_command_encoder(&wgpu::wgt::CommandEncoderDescriptor {
                    label: Some("Create Encoder For Pow"),
                });

        {
            let mut bcp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Create Begin Compute Pass For Pow"),
                timestamp_writes: None,
            });

            bcp.set_pipeline(&pipeline);
            bcp.set_bind_group(0, Some(&heap_bind.binding_groups), &[]);
            bcp.set_bind_group(1, Some(&array_bind.1), &[]);
            bcp.set_bind_group(2, Some(&out_bind.1), &[]);
            bcp.set_bind_group(3, Some(&scalar_bind.1), &[]);

            let x = (len + 255) / 256;
            bcp.dispatch_workgroups(x, 1, 1);
        }

        wgpu.queue.submit(Some(encoder.finish()));

        let array = GpuArray {
            module: Arc::new(self.clone()),
            length: len as usize,
            pointer: (allocate.1, allocate.2),
            binding: out_bind,
            shape: shape.clone(),
            space_type: allocate.0,
            stride,
        };

        Ok(array)
    }

    fn scalar_bind(
        &self,
        device: &Device,
        scalar: f32,
    ) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        let scalar_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Create Scalar Buffer For Pow"),
            contents: bytemuck::bytes_of(&scalar),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Create Scalar Bind Griup Layout For Pow"),
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Create Scalar Bind Griup Layout For Pow"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: scalar_buffer.as_entire_binding(),
            }],
        });

        (bind_group_layout, bind_group)
    }
}
